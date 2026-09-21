"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ...core import register
from .box_ops import batched_generalized_box_iou, box_cxcywh_to_xyxy, generalized_box_iou


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ["use_focal_loss"]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict["cost_class"]
        self.cost_bbox = weight_dict["cost_bbox"]
        self.cost_giou = weight_dict["cost_giou"]

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: dict[str, torch.Tensor], targets, return_topk=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if not return_topk:
            return {"indices": self.match_many([outputs], targets)[0]}

        bs, num_queries = outputs["pred_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            # print("out_prob:", out_prob.shape)
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices_pre
        ]

        # Compute topk indices
        if return_topk:
            return {"indices_o2m": self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    @torch.no_grad()
    def match_many(self, outputs_list, targets):
        """Hungarian-match several prediction sets against the same targets.

        Gives the indices ``forward`` gives for each set, but only computes each
        image's own [num_queries, num_gt] cost block (``forward`` used to build the
        whole [bs * num_queries, total_gt] matrix and throw away the off-diagonal
        blocks), and moves the cost blocks of every set to the host in a single
        copy, i.e. one device sync for all decoder layers instead of one per layer.
        """
        sizes = [len(t["boxes"]) for t in targets]
        bs, max_gt = len(targets), max(sizes, default=0)
        empty = torch.zeros(0, dtype=torch.int64)
        if max_gt == 0:
            return [[(empty, empty) for _ in range(bs)] for _ in outputs_list]

        tgt_ids = torch.nn.utils.rnn.pad_sequence([t["labels"] for t in targets], batch_first=True)
        tgt_bbox = torch.nn.utils.rnn.pad_sequence([t["boxes"] for t in targets], batch_first=True)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        checks = [(tgt_xyxy[..., 2:] >= tgt_xyxy[..., :2]).all()]

        # Sets with the same dtypes and query count are stacked into one batch so
        # each cost term is one kernel for all of them; the arithmetic per element
        # is unchanged, so matching is too.
        groups = {}
        for n, out in enumerate(outputs_list):
            logits, boxes = out["pred_logits"], out["pred_boxes"]
            groups.setdefault((logits.dtype, boxes.dtype, tuple(logits.shape[1:])), []).append(n)

        costs, slots = [], [None] * len(outputs_list)
        offset = 0
        for members in groups.values():
            logits = torch.cat([outputs_list[n]["pred_logits"] for n in members])
            boxes = torch.cat([outputs_list[n]["pred_boxes"] for n in members])
            k = len(members)
            num_queries = logits.shape[1]
            ids = tgt_ids.repeat(k, 1)
            gt_boxes = tgt_bbox.repeat(k, 1, 1)
            gt_xyxy = tgt_xyxy.repeat(k, 1, 1)

            index = ids[:, None, :].expand(-1, num_queries, -1)
            if self.use_focal_loss:
                out_prob = F.sigmoid(logits).gather(2, index)
                neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -logits.softmax(-1).gather(2, index)

            cost_bbox = torch.cdist(boxes, gt_boxes, p=1)
            out_xyxy = box_cxcywh_to_xyxy(boxes)
            checks.append((out_xyxy[..., 2:] >= out_xyxy[..., :2]).all())
            cost_giou = -batched_generalized_box_iou(out_xyxy, gt_xyxy)

            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            # nan_to_num before the (exact) up-cast so +/-inf clamp to the same
            # dtype limits as before
            costs.append(torch.nan_to_num(C, nan=1.0).float().flatten())
            for j, n in enumerate(members):
                slots[n] = (offset, j, num_queries, k)
            offset += costs[-1].numel()

        costs = torch.cat(costs).cpu()
        # Degenerate (x2 < x1) boxes used to trip an assert in generalized_box_iou;
        # the flags ride along with the copy above instead of costing their own syncs.
        assert bool(torch.stack(checks).all().cpu()), "degenerate boxes in matcher input"

        results = []
        for off, j, num_queries, k in slots:
            C = costs[off : off + k * bs * num_queries * max_gt].view(k, bs, num_queries, max_gt)[j]
            per_image = []
            for b, size in enumerate(sizes):
                if size == 0:
                    per_image.append((empty, empty))
                    continue
                i, jj = linear_sum_assignment(C[b, :, :size])
                per_image.append((torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(jj, dtype=torch.int64)))
            results.append(per_image)
        return results

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = (
                [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            )
            indices_list.append(
                [
                    (
                        torch.as_tensor(i, dtype=torch.int64),
                        torch.as_tensor(j, dtype=torch.int64),
                    )
                    for i, j in indices_k
                ]
            )
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6

        indices_list = [
            (
                torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                torch.cat([indices_list[i][j][1] for i in range(k)], dim=0),
            )
            for j in range(len(sizes))
        ]

        # C.copy_(C_original)
        return indices_list
