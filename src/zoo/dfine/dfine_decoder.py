"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

The D-FINE decoder: a deformable-attention decoder stack with Fine-grained Distribution
Refinement, fed by the ``num_queries`` best encoder tokens. Query initialization is the one
method a subclass can override (``_get_decoder_input``). The query count may differ per image: the initial queries
are padded to the largest count and ``batch_queries_num`` tells the criterion and the denoising
mask how many are real.
"""

import copy
from collections import OrderedDict
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.nn.init as init

from ...core import register
from ...misc.visualizer import SAVE_INTERMEDIATE_VISUALIZE_RESULT, dump_boxes
from ...nn.functional import bias_init_with_prob, inverse_sigmoid
from ...nn.transformer import MLP, TransformerDecoderLayer
from .denoising import get_contrastive_denoising_training_group
from .fdr import LQE, Integral, distance2bbox, weighting_function

__all__ = ["DFINETransformer", "TransformerDecoder"]


class TransformerDecoder(nn.Module):
    """
    The decoder stack with Fine-grained Distribution Refinement: every layer predicts a
    correction to the binned edge distributions of the layer before, the boxes are decoded from
    the accumulated distribution around the first layer's reference boxes, and a location
    quality estimator adjusts the class scores. Layers past ``eval_idx`` (used in training only)
    can be ``layer_scale`` times wider.
    """

    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
        fine_dim=0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.fine_dim = fine_dim  # > 0: the first level is the raw fine map, so many channels wide
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max)) for _ in range(num_layers)])

    def value_op(self, feats, value_scale=None):
        """
        The projected levels ``[bs, C, h, w]`` as the deformable cross-attention's values, one
        ``[bs, heads, c, h * w]`` tensor per level: a view of the level, cast to fp32 (the
        precision ``grid_sample`` runs at under autocast) once for every layer, so no layer
        copies or casts them again and the backward pass keeps one copy. ``value_scale`` (the
        wider training-only layers) interpolates the channels to it. A raw fine level
        (``fine_dim``) stays ``[bs, fine_dim, h * w]``, whole for every head and every width.
        """
        values = []
        for level, feat in enumerate(feats):
            bs, _, h, w = feat.shape
            value = feat.flatten(2)  # [bs, C, h * w]
            if level == 0 and self.fine_dim > 0:
                values.append(value.float())
                continue
            if value_scale is not None:
                value = F.interpolate(value.transpose(1, 2), size=value_scale).transpose(1, 2)
            values.append(value.float().reshape(bs, self.num_head, -1, h * w))
        return values

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(
        self,
        target,
        ref_points_unact,
        feats,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        up,
        reg_scale,
        attn_mask=None,
        img_input=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(feats)

        dec_out_bboxes, dec_out_logits, dec_out_pred_corners, dec_out_refs = [], [], [], []
        project = self.project if hasattr(self, "project") else weighting_function(self.reg_max, up, reg_scale)

        ref_points_detach = F.sigmoid(ref_points_unact)
        if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
            dump_boxes("ref_bbox", img_input, ref_points_detach[0])

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # the wider training-only layers work on interpolated queries and values
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(feats, query_pos_embed.shape[-1])
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                # the first layer predicts plain boxes; they anchor the distributions of every layer
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # refine the edge distributions, carrying the previous layer's correction along
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = self.lqe_layers[i](score_head[i](output), pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        if SAVE_INTERMEDIATE_VISUALIZE_RESULT and dec_out_bboxes:
            probs = dec_out_logits[-1][0].softmax(-1)
            dump_boxes("dec_out_bboxes", img_input, dec_out_bboxes[-1][0], probs.argmax(-1), probs.max(-1).values)

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


class DecoderInput(NamedTuple):
    """What ``_get_decoder_input`` hands the decoder (and, in training, the criterion)."""

    contents: torch.Tensor  # [B, Q, D] initial query contents, detached
    boxes_unact: torch.Tensor  # [B, Q, 4] initial boxes as logits, detached
    enc_bboxes_list: list  # encoder-side predictions for the auxiliary loss, one set per entry
    enc_logits_list: list
    batch_queries_num: list  # the real query count of every image (the rest is padding)


@register()
class DFINETransformer(nn.Module):
    """
    Args:
        feat_channels / feat_strides / num_levels: the encoder levels; levels beyond those given
            are made by strided 3x3 convs on the last one.
        num_layers / eval_idx: decoder depth, and the layer whose output is used at inference
            (the later ones train the earlier ones and are dropped by ``convert_to_deploy``).
        num_denoising / label_noise_ratio / box_noise_scale: contrastive denoising queries.
        reg_max / reg_scale: the FDR edge distributions.
        query_select_method: how encoder tokens are ranked, ``default`` (best class score),
            ``one2many`` (every class score) or ``agnostic`` (a single objectness score).
        num_queries: the number of encoder tokens taken as initial queries.
        min_sample_cells: the deformable cross-attention samples within a radius proportional to
            the query's box; with a value above 0 that radius is at least so many cells on every
            level (1: the neighbouring cells), so a tiny box still reads its surroundings on the
            coarse levels (0: off).
        anchor_grid_size: the anchor box of the first level's tokens, as a fraction of the image;
            it doubles per level (D-FINE's 0.05 is 5 cells of a stride-8 first level; a stride-4
            first level wants 0.025 for the same ratio). Must exceed ``eps``, or the border test
            marks every token invalid.
        anchor_cells: with a value above 0, the anchor box of every level's tokens is so many
            cells of that level instead, square in pixels (``anchor_cells * stride``) whatever
            the input's size or aspect. A fraction of the image grows with the input: at 1333x800
            the 0.05 anchors are 67x40 px against the 40x40 of the 800x800 training crops, one
            reason AP collapses above the training extent. 10 reproduces 0.05 on 800x800 inputs.
        fine_channels: with a value above 0, the encoder's ``fine`` map (``HybridEncoder``'s
            fine level, so many channels wide, one stride finer than its first level) is read
            by every decoder layer's deformable cross-attention as its first, finest level,
            raw: each head samples the map at its points and a per-head linear lifts the
            weighted sum of its samples to the head's width (``MSDeformableAttention``'s
            ``fine_dim``). Projecting the samples rather than the map reads the same function
            class (a linear map commutes with bilinear sampling) and keeps the dense map, its
            fp32 copy and the gradient buffers of the sampling's backward pass ``fine_channels``
            wide rather than ``hidden_dim``. It is values only: no token of it enters the query
            selection, gets an anchor or is scored, so the selection and the losses are those of
            the encoder levels alone. ``num_points`` then has one more entry, its first for the
            fine level. 0 (default): no fine level.
        subcell: every token of the finest pyramid level gathers the four stride-2 sub-cells it
            covers before it is scored: a softmax over five logits, one from the token itself and
            one from each sub-cell's fine features, weights the sub-cells' features, and their
            weighted sum is projected and added to the token. The token's own slot carries no
            features, so weight on it means the sub-cells are left out. The score and box heads,
            the budget, the top-k and the anchors are untouched and the token is scored once, so
            the level competes with the others as before; what the sub-cells contribute, and
            which of them, is what the weights learn. The projection starts at zero, so a run
            begins exactly where it begins without this. About 1.4 GFLOPs at 800x800. Needs a
            fine level.
        query_budget: how many encoder tokens become queries. ``fixed``: ``num_queries`` for
            every image (D-FINE). ``threshold``: the number of valid tokens whose best class
            score exceeds ``count_threshold`` plus ``count_margin``, rounded up to a multiple of
            ``count_round`` and clamped to ``count_range``; in training the ground-truth count
            takes the place of the token count (``count_train_budget`` ``gt``) or the larger of
            the two is used (``max``). Nothing to train: the encoder's scores already count. The
            images of a batch are padded to the largest budget; the padded queries are masked out
            of the attention, the losses and the detections.
    """

    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        feat_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32, 64, 128),
        num_levels=5,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        num_queries=300,
        min_sample_cells=0.0,
        anchor_grid_size=0.05,
        anchor_cells=0,
        fine_channels=0,
        subcell=False,
        query_budget="fixed",
        count_train_budget="gt",
        count_threshold=0.4,
        count_margin=200,
        count_round=100,
        count_range=(300, 1500),
    ):
        super().__init__()
        assert query_budget in ("fixed", "threshold"), query_budget
        assert count_train_budget in ("gt", "max"), count_train_budget
        assert len(feat_channels) <= num_levels
        assert anchor_grid_size > eps, f"anchor_grid_size {anchor_grid_size} must exceed eps {eps}"
        assert len(feat_strides) == len(feat_channels)
        assert query_select_method in ("default", "one2many", "agnostic"), query_select_method
        assert cross_attn_method in ("default", "discrete"), cross_attn_method

        feat_strides = list(feat_strides)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.eps = eps
        self.anchor_grid_size = anchor_grid_size
        self.anchor_cells = anchor_cells
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.reg_max = reg_max
        self.num_queries = num_queries
        self.query_budget = query_budget
        self.count_train_budget = count_train_budget
        self.count_threshold = count_threshold
        self.count_margin = count_margin
        self.count_round = count_round
        self.count_range = tuple(count_range)
        if query_budget == "threshold":
            assert count_round > 0 and self.count_range[0] <= self.count_range[1], (count_round, count_range)
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        self.fine_channels = fine_channels
        self.subcell = bool(subcell) and fine_channels > 0
        # the cross-attention's levels: the fine level, if any, then the encoder levels
        self.num_value_levels = num_levels + (1 if fine_channels > 0 else 0)
        if isinstance(num_points, (list, tuple)):
            assert len(num_points) == self.num_value_levels, (
                f"num_points needs one entry per cross-attention level ({self.num_value_levels}, "
                f"the fine level first), got {list(num_points)}"
            )
            num_points = list(num_points)

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # transformer
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        layer_args = dict(
            d_model=hidden_dim,
            n_head=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=self.num_value_levels,
            n_points=num_points,
            cross_attn_method=cross_attn_method,
            min_sample_cells=min_sample_cells,
            fine_dim=fine_channels,
        )
        decoder_layer = TransformerDecoderLayer(**layer_args)
        decoder_layer_wide = TransformerDecoderLayer(**layer_args, layer_scale=layer_scale)
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            fine_dim=fine_channels,
        )

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)

        # encoder output: token scores and boxes for query selection
        self.enc_output = nn.Sequential(
            OrderedDict([("proj", nn.Linear(hidden_dim, hidden_dim)), ("norm", nn.LayerNorm(hidden_dim))])
        )
        self.enc_score_head = nn.Linear(hidden_dim, 1 if query_select_method == "agnostic" else num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        if self.subcell:
            # the attention logits: one from the token, one from each sub-cell's fine features
            # (LayerNorm first, the fine map being a raw convolution output with its own scale)
            self.subcell_token_score = nn.Linear(hidden_dim, 1)
            self.subcell_child_score = nn.Sequential(nn.LayerNorm(fine_channels), nn.Linear(fine_channels, 1))
            # into the token space as enc_output does: normalised first, the token being a LayerNorm output
            self.subcell_proj = nn.Sequential(nn.LayerNorm(fine_channels), nn.Linear(fine_channels, hidden_dim))

        # decoder heads, one per layer; the training-only layers past eval_idx may be wider
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        num_wide = num_layers - self.eval_idx - 1
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, num_classes) for _ in range(num_wide)]
        )
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.dec_bbox_head = nn.ModuleList(
            [MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3) for _ in range(self.eval_idx + 1)]
            + [MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3) for _ in range(num_wide)]
        )
        self.integral = Integral(self.reg_max)

        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.dec_score_head[self.eval_idx]])
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        if self.subcell:
            # the projection starts at zero: the token is itself at first, and the weights learn
            # from the second step, once the projection has moved
            init.constant_(self.subcell_proj[1].weight, 0)
            init.constant_(self.subcell_proj[1].bias, 0)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)

        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _proj(self, in_channels, kernel_size, stride):
        """A conv-BN projection to ``hidden_dim`` (nothing, for a 1x1 from ``hidden_dim``)."""
        if in_channels == self.hidden_dim and kernel_size == 1:
            return nn.Identity()
        conv = nn.Conv2d(in_channels, self.hidden_dim, kernel_size, stride, padding=kernel_size // 2, bias=False)
        return nn.Sequential(OrderedDict([("conv", conv), ("norm", nn.BatchNorm2d(self.hidden_dim))]))

    def _build_input_proj_layer(self, feat_channels):
        """A projection per level; extra levels downsample the last one with stride 2."""
        self.input_proj = nn.ModuleList(self._proj(c, 1, 1) for c in feat_channels)
        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(self._proj(in_channels, 3, 2))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: list[torch.Tensor]):
        """
        The projected encoder levels (the query selection's tokens), and the same flattened to
        ``[b, sum(h*w), c]`` with their ``(h, w)``. The fine level is not among them.
        """
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        for i in range(len(feats), self.num_levels):
            source = feats[-1] if i == len(feats) else proj_feats[-1]
            proj_feats.append(self.input_proj[i](source))

        spatial_shapes = [list(feat.shape[2:]) for feat in proj_feats]
        memory = torch.concat([feat.flatten(2).permute(0, 2, 1) for feat in proj_feats], 1)
        return proj_feats, memory, spatial_shapes

    def _subcell_attend(self, output_memory, fine, spatial_shapes):
        """
        The finest level's tokens gather their four stride-2 sub-cells: a softmax over the token's
        own logit and one per sub-cell weights the sub-cells' fine features, and their weighted sum
        is projected and added to the token. Weight on the token's own slot leaves the sub-cells
        out. Returns ``output_memory`` with the level's tokens updated, the other levels as they were.
        """
        bs, c = fine.shape[:2]
        h0, w0 = (int(x) for x in spatial_shapes[0])
        n0 = h0 * w0
        maps = fine.reshape(bs, c, *fine.shape[2:])
        if maps.shape[2:] != (2 * h0, 2 * w0):
            # the stem map is ceil(H / 2) and the level ceil(H / 4): on a side that is not a
            # multiple of 4 they differ by one row or column, padded here (zero) or cropped
            maps = F.pad(maps, (0, 2 * w0 - maps.shape[3], 0, 2 * h0 - maps.shape[2]))[:, :, : 2 * h0, : 2 * w0]
        maps = maps.permute(0, 2, 3, 1)  # [bs, 2h0, 2w0, C]
        # the sub-cells' logits on the map where it lies (the head is a 1x1), then the four of each
        # token side by side: [bs, h0, w0, 4], top-left, top-right, bottom-left, bottom-right
        child = self.subcell_child_score(maps).squeeze(-1)
        corners = [(dy, dx) for dy in (0, 1) for dx in (0, 1)]
        token = output_memory[:, :n0]
        logits = torch.cat(
            [
                self.subcell_token_score(token).view(bs, h0, w0, 1),
                *(child[:, dy::2, dx::2, None] for dy, dx in corners),
            ],
            dim=-1,
        )
        weights = F.softmax(logits, dim=-1)  # [bs, h0, w0, 5]: the token itself, then its sub-cells
        # the weighted sum over strided views of the map: no stacked copy of the four sub-cells
        gathered = sum(weights[..., 1 + k, None] * maps[:, dy::2, dx::2] for k, (dy, dx) in enumerate(corners))
        return torch.cat([token + self.subcell_proj(gathered.flatten(1, 2)), output_memory[:, n0:]], dim=1)

    def _generate_anchors(self, spatial_shapes=None, dtype=torch.float32, device="cpu"):
        """
        One anchor box (as logits) per token of every level, ``anchor_grid_size * 2 ** level`` of
        the image wide, or ``anchor_cells`` cells of the level; boxes too close to the border are
        marked invalid. Cached per (shapes, device): a forward
        pays for them once per input size. (Built on the host as before: a CUDA division by a
        Python scalar rounds differently and flips border tokens' validity.)
        """
        if spatial_shapes is None:
            eval_h, eval_w = self.eval_spatial_size
            spatial_shapes = [[int(eval_h / s), int(eval_w / s)] for s in self.feat_strides]
        key = (tuple(tuple(int(x) for x in hw) for hw in spatial_shapes), str(device), dtype)
        cache = self.__dict__.setdefault("_anchor_cache", {})  # plain attribute: not a buffer, not saved
        if key in cache:
            return cache[key]
        if len(cache) >= 64:  # variable eval sizes: bound the cache
            cache.clear()

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            if self.anchor_cells > 0:
                wh = torch.ones_like(grid_xy) * self.anchor_cells / torch.tensor([w, h], dtype=dtype)
            else:
                wh = torch.ones_like(grid_xy) * self.anchor_grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, dim=1).to(device, non_blocking=True)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        cache[key] = (anchors, valid_mask)
        return anchors, valid_mask

    # ------------------------------------------------------------------ query initialization

    def _select_topk(self, memory, outputs_logits, outputs_anchors_unact, topk, return_index=False):
        """The ``topk`` tokens by ``query_select_method``, with their logits and anchors."""
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        else:  # agnostic
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)

        def gather(x):
            return x.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, x.shape[-1]))

        if return_index:
            return gather(memory), gather(outputs_logits), gather(outputs_anchors_unact), topk_ind
        return gather(memory), gather(outputs_logits), gather(outputs_anchors_unact)

    def _threshold_budgets(self, token_count, targets):
        """
        Per image, the number of queries under ``query_budget='threshold'``: the count (of tokens
        above the threshold, or in training the ground truth's, or the larger of the two) plus
        ``count_margin``, rounded up to ``count_round`` and clamped to ``count_range``. On the host.
        """
        count = token_count
        if self.training and targets is not None:
            gt = torch.tensor([len(t["labels"]) for t in targets]).to(count.device, non_blocking=True)
            count = gt if self.count_train_budget == "gt" else torch.maximum(count, gt)
        lo, hi = self.count_range
        budget = torch.div(count + self.count_margin + self.count_round - 1, self.count_round, rounding_mode="floor")
        return (budget * self.count_round).clamp(lo, hi).tolist()

    def _get_decoder_input(self, memory, spatial_shapes, encoder_out, targets=None):
        """
        The initial queries: the best encoder tokens, ``num_queries`` of them or the image's
        budget (``query_budget``). Returns their contents and boxes (as logits, both detached,
        padded to the largest count in the batch), the encoder-side predictions for the
        auxiliary loss and the real query count of every image, as a ``DecoderInput``. A subclass may choose its
        queries otherwise; ``targets`` (training only) may take part in the choice.
        """
        anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        b = memory.shape[0]
        if b > 1:
            anchors = anchors.repeat(b, 1, 1)
        memory = valid_mask.to(memory.dtype) * memory

        output_memory: torch.Tensor = self.enc_output(memory)
        if self.subcell:  # the finest level's tokens gather their sub-cells before they are scored
            output_memory = self._subcell_attend(output_memory, encoder_out["fine"], spatial_shapes)
        enc_outputs_logits: torch.Tensor = self.enc_score_head(output_memory)

        if self.query_budget == "fixed":
            batch_queries_num = [self.num_queries] * b
        else:
            # every valid token's best class score; the budget counts the ones above the threshold
            scores = F.sigmoid(
                enc_outputs_logits.squeeze(-1)
                if self.query_select_method == "agnostic"
                else enc_outputs_logits.max(-1).values
            )
            scores = scores.masked_fill(~valid_mask[..., 0].expand(b, -1), 0.0)
            batch_queries_num = self._threshold_budgets((scores > self.count_threshold).sum(1), targets)
        batch_queries_num = [
            min(n, enc_outputs_logits.shape[1]) for n in batch_queries_num
        ]  # never more than the candidates
        k = max(batch_queries_num)
        topk_memory, topk_logits, topk_anchors = self._select_topk(output_memory, enc_outputs_logits, anchors, k)
        topk_bbox_unact = self.enc_bbox_head(topk_memory) + topk_anchors
        if min(batch_queries_num) < k:
            # the images with a smaller budget are padded: their surplus tokens are zeroed here and
            # masked everywhere else by batch_queries_num
            counts = torch.tensor(batch_queries_num, device=memory.device)
            pad = (torch.arange(k, device=memory.device)[None, :] >= counts[:, None])[..., None]
            topk_memory = topk_memory.masked_fill(pad, 0.0)
            topk_logits = topk_logits.masked_fill(pad, 0.0)
            topk_bbox_unact = topk_bbox_unact.masked_fill(pad, 0.0)

        enc_topk_bboxes_list = [F.sigmoid(topk_bbox_unact)]
        enc_topk_logits_list = [topk_logits]
        return DecoderInput(
            topk_memory.detach(),
            topk_bbox_unact.detach(),
            enc_topk_bboxes_list,
            enc_topk_logits_list,
            batch_queries_num,
        )

    # ------------------------------------------------------------------ forward

    def forward(self, encoder_out, targets=None):
        feats = encoder_out["feats"]
        img_inputs = encoder_out["img_inputs"]

        proj_feats, memory, spatial_shapes = self._get_encoder_input(feats)
        # the cross-attention's levels: the raw fine level first, if any; the selection, the
        # anchors and the FDR unit below stay on the encoder levels
        value_feats, value_shapes = proj_feats, spatial_shapes
        if self.fine_channels > 0:
            fine = encoder_out["fine"]
            value_feats, value_shapes = [fine, *proj_feats], [list(fine.shape[2:]), *spatial_shapes]

        dec_in = self._get_decoder_input(memory, spatial_shapes, encoder_out, targets)
        init_ref_contents, init_ref_points_unact = dec_in.contents, dec_in.boxes_unact
        enc_topk_bboxes_list, enc_topk_logits_list = dec_in.enc_bboxes_list, dec_in.enc_logits_list
        batch_queries_num = dec_in.batch_queries_num
        num_queries = max(batch_queries_num)

        # denoising queries are prepended to the matching queries during training
        dn_meta = None
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = get_contrastive_denoising_training_group(
                targets,
                self.num_classes,
                num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                batch_queries_num=batch_queries_num,
            )
            init_ref_points_unact = torch.concat([denoising_bbox_unact, init_ref_points_unact], dim=1)
            init_ref_contents = torch.concat([denoising_logits, init_ref_contents], dim=1)
        else:
            attn_mask = self._padding_attn_mask(batch_queries_num, memory.device)
        if attn_mask is not None:
            # the additive mask scaled_dot_product_attention takes, built once for every layer in
            # the attention's dtype (a boolean one would be converted in every layer, slower)
            dtype = torch.get_autocast_dtype(memory.device.type) if torch.is_autocast_enabled() else torch.float32
            attn_mask = torch.zeros(attn_mask.shape, dtype=dtype, device=memory.device).masked_fill(
                attn_mask, -torch.inf
            )
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            value_feats,
            value_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            self.pre_bbox_head,
            self.integral,
            self.up,
            self.reg_scale,
            attn_mask=attn_mask,
            img_input=img_inputs,
        )

        if dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)

        if min(batch_queries_num) < num_queries:
            # padded queries never become detections (the criterion masks them by count anyway)
            counts = torch.tensor(batch_queries_num).to(memory.device, non_blocking=True)
            pad = torch.arange(num_queries, device=memory.device)[None, :] >= counts[:, None]
            out_logits = out_logits.masked_fill(pad[None, :, :, None], -1e4)
            pre_logits = pre_logits.masked_fill(pad[:, :, None], -1e4)

        out = {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}
        if self.training:
            out["pred_corners"] = out_corners[-1]
            out["ref_points"] = out_refs[-1]
            out["up"] = self.up
            out["reg_scale"] = self.reg_scale

        if self.training and self.aux_loss:
            out["aux_outputs"] = self._layer_outputs(
                out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1], out_corners[-1], out_logits[-1]
            )
            out["enc_aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b} for a, b in zip(enc_topk_logits_list, enc_topk_bboxes_list)
            ]
            out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
            out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

            if dn_meta is not None:
                out["dn_outputs"] = self._layer_outputs(
                    dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs, dn_out_corners[-1], dn_out_logits[-1]
                )
                out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                out["dn_meta"] = dn_meta

        for key, value in encoder_out.items():
            if key not in ("feats", "fine"):
                out[key] = value
        out["batch_queries_num"] = batch_queries_num
        return out

    def _padding_attn_mask(self, batch_queries_num, device):
        """
        Without denoising queries: a ``[B, 1, Q, Q]`` self-attention mask (True blocks, the same
        for every head) hiding every image's padded queries from its real ones and vice versa
        (padding still sees itself), or ``None`` when no image is padded.
        """
        num_queries = max(batch_queries_num)
        if min(batch_queries_num) == num_queries:
            return None
        counts = torch.tensor(batch_queries_num).to(device, non_blocking=True)  # no stream sync
        real = torch.arange(num_queries, device=device)[None, :] < counts[:, None]  # [B, Q]
        return (real[:, :, None] != real[:, None, :])[:, None]

    @staticmethod
    @torch.jit.unused
    def _layer_outputs(logits, boxes, corners, refs, teacher_corners=None, teacher_logits=None):
        """One prediction dict per decoder layer, for the auxiliary and denoising losses."""
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_corners": c,
                "ref_points": d,
                "teacher_corners": teacher_corners,
                "teacher_logits": teacher_logits,
            }
            for a, b, c, d in zip(logits, boxes, corners, refs)
        ]
