"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from contextlib import contextmanager
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from ...core import register
from ...misc import dist_utils
from ...misc.box_ops import box_cxcywh_to_xyxy, elementwise_box_iou, elementwise_generalized_box_iou
from ...solver.matcher import FlatMatches, PaddedTargets, padded_targets
from .fdr import bbox2distance

__all__ = ["DomeCriterion"]


Targets = PaddedTargets  # a batch's ground truths padded to [B, M], see matcher.padded_targets


class Pairs(NamedTuple):
    """
    The matched pairs of a set stack as flat index tensors, sorted by set: pair ``k`` matches
    query ``query_idx[k]`` of image ``batch_idx[k]`` in set ``set_idx[k]`` to that image's
    ground truth ``target_idx[k]`` (an index into the padded targets). ``counts`` is the number
    of pairs of every set, on the host.
    """

    set_idx: Tensor
    batch_idx: Tensor
    query_idx: Tensor
    target_idx: Tensor
    counts: list[int]

    @classmethod
    def from_lists(cls, indices_lists, device):
        """From one per-image list of ``(pred_idx, target_idx)`` per set, as the matcher returns them."""
        lengths = [[src.shape[0] for src, _ in indices] for indices in indices_lists]
        counts = [sum(lens) for lens in lengths]
        set_idx = torch.arange(len(indices_lists)).repeat_interleave(torch.tensor(counts, dtype=torch.long))
        batch_idx = torch.cat(
            [torch.arange(len(lens)).repeat_interleave(torch.tensor(lens, dtype=torch.long)) for lens in lengths]
        )
        query_idx = torch.cat([src for indices in indices_lists for src, _ in indices])
        target_idx = torch.cat([tgt for indices in indices_lists for _, tgt in indices])
        return cls(
            set_idx.to(device, non_blocking=True),
            batch_idx.to(device, non_blocking=True),
            query_idx,
            target_idx,
            counts,
        )

    @classmethod
    def from_flat(cls, flat: FlatMatches):
        """From the matcher's flat matches of one or more sets (sorted by set and image already)."""
        device = flat.query_idx.device
        set_idx = torch.arange(len(flat.counts)).repeat_interleave(torch.tensor(flat.counts, dtype=torch.long))
        return cls(
            set_idx.to(device, non_blocking=True), flat.batch_idx, flat.query_idx, flat.target_idx, list(flat.counts)
        )

    @classmethod
    def shared(cls, indices, num_sets, device):
        """The same per-image matching for every one of ``num_sets`` sets."""
        return cls.from_lists([indices], device).tiled(num_sets)

    def tiled(self, num_sets):
        """These pairs of one set, as the same matching for every one of ``num_sets`` sets."""
        k = self.counts[0]
        set_idx = torch.arange(num_sets, device=self.query_idx.device).repeat_interleave(k)
        return Pairs(
            set_idx,
            self.batch_idx.repeat(num_sets),
            self.query_idx.repeat(num_sets),
            self.target_idx.repeat(num_sets),
            [k] * num_sets,
        )

    def first_sets(self, n):
        """The pairs of the first ``n`` sets."""
        k = sum(self.counts[:n])
        return Pairs(self.set_idx[:k], self.batch_idx[:k], self.query_idx[:k], self.target_idx[:k], self.counts[:n])


class SetStack(NamedTuple):
    """
    Prediction sets of the same shape stacked along a leading set dimension, so that one loss
    computation covers them all: ``logits [S, B, Q, C]`` and ``boxes [S, B, Q, 4]`` of every
    set; ``corners [S', B, Q, 4 * (reg_max + 1)]`` and ``refs [S', B, Q, 4]`` of the first ``S'``
    sets, which carry FDR's edge distributions (``None`` when none does); the distillation
    teacher of those sets, if any, with ``has_teacher`` / ``is_teacher`` per set (a set is its own
    teacher in the denoising stack; its distillation loss is zero); the loss suffix per set;
    ``q_valid`` (``None``: every query is real); and whether the sets are denoising ones.
    """

    logits: Tensor
    boxes: Tensor
    corners: Tensor | None
    refs: Tensor | None
    teacher_corners: Tensor | None
    teacher_logits: Tensor | None
    has_teacher: list[bool]
    is_teacher: list[bool]
    suffixes: list[str]
    q_valid: Tensor | None
    is_dn: bool

    @classmethod
    def build(cls, sets, suffixes, q_valid, is_dn=False):
        logits = torch.stack([s["pred_logits"] for s in sets])
        boxes = torch.stack([s["pred_boxes"] for s in sets])
        with_corners = [s for s in sets if "pred_corners" in s]
        assert with_corners == sets[: len(with_corners)], "the sets with edge distributions come first"
        corners = refs = teacher_corners = teacher_logits = None
        has_teacher, is_teacher = [], []
        if with_corners:
            corners = torch.stack([s["pred_corners"] for s in with_corners])
            refs = torch.stack([s["ref_points"] for s in with_corners]).detach()
            teacher_corners = next(
                (s["teacher_corners"] for s in with_corners if s.get("teacher_corners") is not None), None
            )
            teacher_logits = next(
                (s["teacher_logits"] for s in with_corners if s.get("teacher_logits") is not None), None
            )
            has_teacher = [s.get("teacher_corners") is not None for s in with_corners]
            is_teacher = [
                teacher_corners is not None and s["pred_corners"].data_ptr() == teacher_corners.data_ptr()
                for s in with_corners
            ]
        return cls(
            logits,
            boxes,
            corners,
            refs,
            teacher_corners,
            teacher_logits,
            has_teacher,
            is_teacher,
            suffixes,
            q_valid,
            is_dn,
        )

    @property
    def num_sets(self):
        return self.logits.shape[0]

    @property
    def num_corner_sets(self):
        return 0 if self.corners is None else self.corners.shape[0]


def _flat_index(shape, *indices: Tensor) -> Tensor:
    """
    The flat (row-major) index into a tensor of ``shape`` of the entries ``indices``, one index
    tensor per dim. For ``index_fill_`` on the flat view: ``x[i, j] = scalar`` (an index put of a
    Python scalar) syncs the host, ``x.view(-1).index_fill_(0, flat, scalar)`` does not.
    """
    assert len(indices) == len(shape), (len(indices), shape)
    flat = indices[0]
    for size, idx in zip(shape[1:], indices[1:]):
        flat = flat * size + idx
    return flat


def _per_set_sum(values: Tensor, counts: list[int]) -> Tensor:
    """The sum of ``values`` (one per pair, sorted by set) within each set, ``[S]``."""
    if len(set(counts)) == 1:
        return values.view(len(counts), -1).sum(1)
    return torch.stack([v.sum() for v in values.split(counts)])


@register()
class DomeCriterion(nn.Module):
    """
    The training loss: D-FINE's set-prediction losses on every prediction set the decoder returns.

    The decoder output carries, besides the last layer's predictions, ``aux_outputs`` (the other
    layers), ``pre_outputs`` (the first layer's plain boxes), ``enc_aux_outputs`` (the encoder
    tokens picked as queries), and their denoising twins ``dn_outputs`` / ``dn_pre_outputs``. Each
    set is matched to the targets and scored with the same ``losses``, and the weighted terms are returned with a suffix naming the set
    (``_aux_0``, ``_pre``, ``_enc_0``, ``_dn_0``, ...). Padded queries (``batch_queries_num``)
    are never matched and count in no loss.

    Sets of one kind (the decoder layers with the pre-outputs, the encoder sets, the denoising
    sets) are stacked (``SetStack``) and their matches flattened (``Pairs``), so every loss runs
    once over a whole stack and reduces per set.

    Args:
        matcher: the Hungarian matcher (injected from the config).
        weight_dict: weight per loss term; terms not listed are dropped.
        losses: which of ``vfl`` (classification, the target score the matched pair's IoU),
            ``boxes`` (L1 + GIoU) and ``local`` (FDR's fine-grained localization and distillation
            losses) to compute.
        alpha, gamma: the focal parameters of the VFL loss.
        reg_max: the FDR bin count of the decoder.
        use_uni_set: match the box and localization losses against the union of the matches of
            every prediction set (D-FINE's 'go' indices) rather than each set's own.
    """

    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        use_uni_set=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma
        self.reg_max = reg_max
        self.use_uni_set = use_uni_set
        self._clear_cache()

    def _clear_cache(self):
        # per-forward caches: the DDF normalisers of the decoder stack, reused by the denoising
        # stack, and the matched pairs' boxes and quality, gathered once per (stack, pairs)
        self.num_pos, self.num_neg = None, None
        self.matched = {}

    # ------------------------------------------------------------------ matched pairs

    def _matched(self, stack: SetStack, pairs: Pairs, targets: Targets):
        """The matched predictions' boxes ``[K, 4]``, the target boxes they are matched to, and their IoU (detached)."""
        key = (id(stack), id(pairs))
        if key not in self.matched:
            _, b, q, _ = stack.boxes.shape
            src_boxes = stack.boxes.reshape(-1, 4)[(pairs.set_idx * b + pairs.batch_idx) * q + pairs.query_idx]
            target_boxes = targets.boxes.reshape(-1, 4)[pairs.batch_idx * targets.boxes.shape[1] + pairs.target_idx]
            with torch.no_grad():
                quality = elementwise_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))[0]
            self.matched[key] = (src_boxes, target_boxes, quality)
        return self.matched[key]

    def _class_targets(self, stack: SetStack, pairs: Pairs, targets: Targets):
        """Per (set, query) target class ``[S, B, Q]`` (``num_classes`` = background) and its one-hot over the real classes."""
        classes = torch.full(stack.logits.shape[:3], self.num_classes, dtype=torch.int64, device=stack.logits.device)
        classes[pairs.set_idx, pairs.batch_idx, pairs.query_idx] = targets.labels[pairs.batch_idx, pairs.target_idx]
        one_hot = F.one_hot(classes, num_classes=self.num_classes + 1)[..., :-1]
        return classes, one_hot

    def _reduce_query_loss(self, loss: Tensor, stack: SetStack, num_boxes):
        """Sum a ``[S, B, Q, C]`` per-query loss over each set, ignoring the padded queries, normalized by ``num_boxes``."""
        if stack.q_valid is not None:
            loss = loss * stack.q_valid[None, :, :, None]
        return loss.sum((1, 2, 3)) / num_boxes

    # ------------------------------------------------------------------ classification losses

    def _iou_aware_targets(self, stack, pairs, targets):
        """The one-hot targets with the matched pair's IoU as the positive score."""
        logits = stack.logits
        _, _, quality = self._matched(stack, pairs, targets)
        classes, target = self._class_targets(stack, pairs, targets)
        target_score = torch.zeros_like(classes, dtype=logits.dtype)
        target_score[pairs.set_idx, pairs.batch_idx, pairs.query_idx] = quality.to(target_score.dtype)
        return logits, target, target_score.unsqueeze(-1) * target

    def loss_labels_vfl(self, stack, pairs, num_boxes, targets, **kwargs):
        src_logits, target, target_score = self._iou_aware_targets(stack, pairs, targets)
        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        return {"loss_vfl": self._reduce_query_loss(loss, stack, num_boxes)}

    # ------------------------------------------------------------------ box losses

    def loss_boxes(self, stack, pairs, num_boxes, targets, **kwargs):
        """L1 and GIoU losses of the matched pairs (boxes are normalized cxcywh)."""
        src_boxes, target_boxes, _ = self._matched(stack, pairs, targets)
        loss_bbox = _per_set_sum((src_boxes - target_boxes).abs().sum(-1), pairs.counts) / num_boxes
        loss_giou = 1 - elementwise_generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        return {"loss_bbox": loss_bbox, "loss_giou": _per_set_sum(loss_giou, pairs.counts) / num_boxes}

    def loss_local(self, stack, pairs, num_boxes, targets, fdr, T=5, **kwargs):  # noqa: N803
        """
        FDR's Fine-Grained Localization (FGL) loss on the matched pairs' edge distributions of
        the sets that have them, and, for those with a distillation teacher, the Decoupled
        Distillation Focal (DDF) loss towards it. ``pairs`` are those sets' pairs.
        """
        if stack.corners is None:
            return {}
        s, b, q = pairs.set_idx, pairs.batch_idx, pairs.query_idx
        src_boxes, target_boxes, quality = self._matched(stack, pairs, targets)
        pred_corners = stack.corners[s, b, q].reshape(-1, self.reg_max + 1)  # [K * 4, bins]
        with torch.no_grad():
            target_corners, weight_right, weight_left = bbox2distance(
                stack.refs[s, b, q],
                box_cxcywh_to_xyxy(target_boxes),
                self.reg_max,
                fdr["reg_scale"],
                fdr["up"],
            )
        weight_targets = quality.unsqueeze(-1).expand(-1, 4).reshape(-1)
        fgl = self.unimodal_distribution_focal_loss(
            pred_corners, target_corners, weight_right, weight_left, weight_targets
        )
        losses = {"loss_fgl": _per_set_sum(fgl, [c * 4 for c in pairs.counts]) / num_boxes}
        if stack.teacher_corners is not None:
            losses["loss_ddf"] = self._loss_ddf(stack, pairs, quality, T)
        return losses

    def _loss_ddf(self, stack: SetStack, pairs: Pairs, quality: Tensor, T):  # noqa: N803
        """KL distillation of every query's edge distributions towards the teacher's, at temperature ``T``, ``[S']``."""
        num_sets, b, q = stack.corners.shape[:3]
        bins = self.reg_max + 1  # explicit, so a set with no query (a batch without a box) still reshapes
        pred_corners = stack.corners.reshape(num_sets, b, q, 4, bins)
        target_corners = stack.teacher_corners.detach().reshape(b, q, 4, bins)

        # matched queries are weighted by their quality, the others by the teacher's confidence
        weight = stack.teacher_logits.sigmoid().max(dim=-1)[0].expand(num_sets, -1, -1).clone()
        weight[pairs.set_idx, pairs.batch_idx, pairs.query_idx] = quality.to(weight.dtype)
        flat = _flat_index((num_sets, b, q), pairs.set_idx, pairs.batch_idx, pairs.query_idx)
        mask = torch.zeros(num_sets * b * q, dtype=torch.bool, device=weight.device).index_fill_(0, flat, True)
        mask = mask.view(num_sets, b, q)
        weight, mask = weight[..., None].expand(-1, -1, -1, 4).detach(), mask[..., None].expand(-1, -1, -1, 4)

        kl = nn.KLDivLoss(reduction="none")(
            F.log_softmax(pred_corners / T, dim=-1), F.softmax(target_corners / T, dim=-1)
        ).sum(-1)
        loss_match_local = weight * (T**2) * kl  # [S', B, Q, 4]

        if not stack.is_dn:
            # balance the matched and unmatched halves; sqrt-scaled so that the GPU batch size does not matter
            batch_scale = 8 / b
            self.num_pos = (mask.sum((1, 2, 3)) * batch_scale) ** 0.5
            self.num_neg = ((~mask).sum((1, 2, 3)) * batch_scale) ** 0.5
        # the halves' means, 0 for an empty half, without reading the masks on the host
        loss_pos = (loss_match_local * mask).sum((1, 2, 3)) / mask.sum((1, 2, 3)).clamp(min=1)
        loss_neg = (loss_match_local * ~mask).sum((1, 2, 3)) / (~mask).sum((1, 2, 3)).clamp(min=1)
        loss = (loss_pos * self.num_pos + loss_neg * self.num_neg) / (self.num_pos + self.num_neg)
        # a set that is its own teacher (the last denoising layer) distils nothing
        own = torch.tensor([0.0 if t else 1.0 for t in stack.is_teacher]).to(loss.device, non_blocking=True)
        return loss * own

    @staticmethod
    def unimodal_distribution_focal_loss(pred, label, weight_right, weight_left, weight=None):
        """Cross-entropy against the two bins around each target position, weighted by their distance to it, per element."""
        dis_left = label.long()
        dis_right = dis_left + 1
        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(-1)
        loss = loss + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)
        if weight is not None:
            loss = loss * weight.float()
        return loss

    # ------------------------------------------------------------------ assembling

    def get_loss(self, loss, stack, pairs, num_boxes, targets, **kwargs):
        loss_map = {"boxes": self.loss_boxes, "vfl": self.loss_labels_vfl, "local": self.loss_local}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](stack, pairs, num_boxes, targets, **kwargs)

    def _stack_losses(self, stack, targets, losses, own, shared, num_boxes, num_shared, uni_losses, fdr=None):
        """
        Every loss in ``losses`` over a stack, weighted and suffixed per set. Losses named in
        ``uni_losses`` use the union matching ``shared`` and its pair count ``num_shared``
        instead of the sets' ``own`` pairs and ``num_boxes`` (a float, or one per set).
        """
        result = {}
        local_pairs = {}
        for loss in losses:
            pairs, nb = (shared, num_shared) if (self.use_uni_set and loss in uni_losses) else (own, num_boxes)
            if loss == "local":  # the sets with edge distributions come first
                if id(pairs) not in local_pairs:
                    local_pairs[id(pairs)] = pairs.first_sets(stack.num_corner_sets)
                pairs = local_pairs[id(pairs)]
            per_set = self.get_loss(loss, stack, pairs, nb, targets, fdr=fdr)
            for k, v in per_set.items():
                if k not in self.weight_dict:
                    continue
                v = v * self.weight_dict[k]  # the sets at once; the entries below are views of it
                for s, suffix in enumerate(stack.suffixes[: v.shape[0]]):
                    if k == "loss_ddf" and not stack.has_teacher[s]:
                        continue
                    result[k + suffix] = v[s]
        return result

    @staticmethod
    def _average_over_ranks(count, device) -> float:
        """A count averaged over the distributed ranks, at least 1."""
        if not dist_utils.is_dist_available_and_initialized():
            return float(max(count, 1))
        count = torch.as_tensor([count], dtype=torch.float, device=device)
        torch.distributed.all_reduce(count)
        return torch.clamp(count / dist_utils.get_world_size(), min=1).item()

    @staticmethod
    def _union_matches(matches: list[FlatMatches], num_queries, num_targets) -> FlatMatches:
        """
        D-FINE's 'go' matching: the union of the matches of several prediction sets, keeping for
        every query the target it was matched to most often (the lowest index on a tie), as flat
        matches of one set sorted by image. Two host syncs.
        """
        b, m = len(num_targets), max(num_targets)
        device = matches[0].query_idx.device
        empty = torch.zeros(0, dtype=torch.long, device=device)
        if m == 0:
            return FlatMatches(empty, empty, empty, [0])
        # every match as one key (image, query, target); the number of sets a key appears in
        key = torch.cat([(f.batch_idx * num_queries + f.query_idx) * m + f.target_idx for f in matches])
        key, count = torch.unique(key, return_counts=True)  # sorted: image, query, then target
        query, target = key // m, key % m  # query numbered across the batch
        best = torch.zeros((b * num_queries,), dtype=count.dtype, device=device)
        best.scatter_reduce_(0, query, count, "amax")
        top = count == best[query]
        first = torch.full((b * num_queries,), m, dtype=torch.long, device=device)
        first.scatter_reduce_(0, query[top], target[top], "amin")  # the lowest target among the ties
        keep = top & (target == first[query])
        query, target = query[keep], target[keep]
        return FlatMatches(query // num_queries, query % num_queries, target, [query.shape[0]])

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """Every denoising query is matched to the ground truth it was made from, group after group."""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        device = targets[0]["labels"].device
        dn_match_indices = []
        for i, t in enumerate(targets):
            num_gt = len(t["labels"])
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device).tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                empty = torch.zeros(0, dtype=torch.int64, device=device)
                dn_match_indices.append((empty, empty))
        return dn_match_indices

    @contextmanager
    def _class_agnostic(self, targets: Targets):
        """Score against a single class: every label becomes 0 and ``num_classes`` is 1 for the duration."""
        num_classes = self.num_classes
        self.num_classes = 1
        try:
            yield targets._replace(labels=torch.zeros_like(targets.labels))
        finally:
            self.num_classes = num_classes

    def forward(self, outputs, targets, **kwargs):
        assert "aux_outputs" in outputs, "DomeCriterion needs the decoder's auxiliary outputs (aux_loss: True)"
        device = outputs["pred_logits"].device
        batch_queries_num = outputs.get("batch_queries_num")
        num_queries = outputs["pred_logits"].shape[1]
        self._clear_cache()
        padded = padded_targets(targets, num_queries, batch_queries_num)
        fdr = {"up": outputs.get("up"), "reg_scale": outputs.get("reg_scale")}

        # match every prediction set in one go, and build the union matching for the box losses;
        # the matches stay flat (one index tensor each) from the matcher to the losses
        sets = [outputs, *outputs["aux_outputs"], outputs["pre_outputs"]]
        suffixes = ["", *(f"_aux_{i}" for i in range(len(outputs["aux_outputs"]))), "_pre"]
        enc_sets = outputs["enc_aux_outputs"]
        matched, _ = self.matcher.match_sets_flat(sets + enc_sets, padded)
        split = sum(matched.counts[: len(sets)])  # the decoder sets' matches first, then the encoder sets'
        dec_matches = FlatMatches(*(t[:split] for t in matched[:3]), matched.counts[: len(sets)])
        enc_matches = FlatMatches(*(t[split:] for t in matched[:3]), matched.counts[len(sets) :])
        own = Pairs.from_flat(dec_matches)
        enc_own = Pairs.from_flat(enc_matches)
        union = self._union_matches([dec_matches, enc_matches], num_queries, padded.num_gt)
        num_go = self._average_over_ranks(union.counts[0], device)
        num_boxes = self._average_over_ranks(sum(padded.num_gt), device)

        # the decoder's sets: their own matches for the classification losses, the union for the boxes
        stack = SetStack.build(sets, suffixes, padded.q_valid)
        shared = Pairs.from_flat(union).tiled(stack.num_sets)
        losses = self._stack_losses(stack, padded, self.losses, own, shared, num_boxes, num_go, ("boxes", "local"), fdr)

        # the encoder sets: their own matches for the classification loss, the union for the boxes
        enc_stack = SetStack.build(enc_sets, [f"_enc_{i}" for i in range(len(enc_sets))], padded.q_valid)
        enc_args = (enc_own, Pairs.from_flat(union).tiled(enc_stack.num_sets), num_boxes, num_go, ("boxes",))
        if outputs["enc_meta"]["class_agnostic"]:
            with self._class_agnostic(padded) as enc_targets:
                losses.update(self._stack_losses(enc_stack, enc_targets, self.losses, *enc_args, fdr))
        else:
            losses.update(self._stack_losses(enc_stack, padded, self.losses, *enc_args, fdr))

        if "dn_outputs" in outputs:
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_sets = [*outputs["dn_outputs"], outputs["dn_pre_outputs"]]
            dn_suffixes = [*(f"_dn_{i}" for i in range(len(outputs["dn_outputs"]))), "_dn_pre"]
            dn_stack = SetStack.build(dn_sets, dn_suffixes, None, is_dn=True)
            dn_pairs = Pairs.shared(indices_dn, dn_stack.num_sets, device)
            # a batch without a box has no denoising groups; the sets are empty and their losses
            # zero, the normaliser only has to stay finite
            dn_num_boxes = num_boxes * max(outputs["dn_meta"]["dn_num_group"], 1)
            losses.update(
                self._stack_losses(dn_stack, padded, self.losses, dn_pairs, dn_pairs, dn_num_boxes, None, (), fdr)
            )

        # a NaN term must not take the whole step down with it: every term cleaned in one kernel,
        # the dict's entries views of the result (``det_engine`` sums them with one more)
        keys = list(losses)
        values = torch.nan_to_num(torch.stack([losses[k] for k in keys]), nan=0.0)
        return {k: values[i] for i, k in enumerate(keys)}
