"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.visualize_src_flatten import visualize_src_flatten

from .utils import get_activation


SAVE_INTERMEDIATE_VISUALIZE_RESULT = os.environ.get("SAVE_INTERMEDIATE_VISUALIZE_RESULT", "False") == "True"
print(f"SAVE_INTERMEDIATE_VISUALIZE_RESULT: {SAVE_INTERMEDIATE_VISUALIZE_RESULT}")


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        # need_weights=False: the averaged attention map was discarded anyway, and
        # not asking for it lets PyTorch use the fused (flash / memory-efficient)
        # kernel instead of materialising [B * heads, L, L] probabilities.
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask, need_weights=False)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


# transformer
class AxisPermutedEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, q, k, v, src_mask=None) -> torch.Tensor:
        src = residual = v
        if self.normalize_before:
            src = self.norm1(src)

        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask, need_weights=False)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class AxisPermutedEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, glob_pos_embeds=None, across_mask=None) -> torch.Tensor:
        """``across_mask`` ([N, N] bool, True = blocked) restricts the second,
        across-window attention; it lets windows of several images share one call."""
        output = src
        B, C, _, _ = glob_pos_embeds.shape
        glob_pos_embeds = glob_pos_embeds.reshape(B, C, -1).permute(0, 2, 1)
        pos_embed = pos_embed.reshape(C, -1).permute(1, 0).unsqueeze(0)
        pos = glob_pos_embeds + pos_embed
        pos_across = pos.permute(1, 0, 2).contiguous()
        for layer in self.layers:
            q = k = self.with_pos_embed(output, pos)
            output = layer(q, k, output, src_mask=src_mask)

            output = output.permute(1, 0, 2).contiguous()
            q = k = self.with_pos_embed(output, pos_across)
            output = layer(q, k, output, src_mask=src_mask if across_mask is None else across_mask)
            output = output.permute(1, 0, 2).contiguous()

        if self.norm is not None:
            output = self.norm(output)

        return output


class WindowProcessor(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        dim_feedforward=1024,
        num_layers=1,
        dropout=0.0,
        activation="relu",
    ):
        """
        窗口处理类
        Args:
            embed_dim: 位置编码维度
            use_residual: 是否使用残差连接
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 相对位置编码器
        self.rel_pos_encoder = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, embed_dim))

        encoder_layer = AxisPermutedEncoderLayer(
            self.embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.window_encoder = AxisPermutedEncoder(encoder_layer, self.num_layers)
        self._coords_cache = {}

    def forward(self, backbone_memory, defe_feature_filtered, window_size, glob_pos_embed):
        """
        前向传播
        Args:
            backbone_memory: 原特征图 [B, C, H, W]
            defe_feature_filtered: 二值图 [B, 1, H, W]
            window_size: 窗口大小 (像素/特征点数)
            feature_enhancer: 可选的特征增强模块
        """

        B, C, H, W = backbone_memory.shape

        try:
            assert H % window_size == 0 and W % window_size == 0
        except AssertionError as e:
            print(f"H and W must be divisible by window_size, but got H: {H}, W: {W}, window_size: {window_size}")
            raise e

        num_win_h = H // window_size
        num_win_w = W // window_size

        rel_pos_embed = self._get_rel_embedding((window_size, window_size)).to(backbone_memory.device)

        # 生成窗口划分
        windows, defe_mask = self._prepare_windows(backbone_memory, defe_feature_filtered, window_size)

        # import pdb; pdb.set_trace()

        if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
            window_mask_vis = defe_mask.float().unsqueeze(-1)  # [B, num_win_h, num_win_w, 1]
            visualize_src_flatten(window_mask_vis, [(num_win_h, num_win_w)], "defe_window_mask", False)

        # All valid windows of the whole batch, image by image in row-major order
        # (the order the per-image loop visited them in). They go through the
        # encoder together; `across_mask` keeps the across-window attention
        # inside each image, so every window sees exactly what it saw when the
        # images were processed one at a time -- without a Python loop over
        # windows and the host syncs its tensor-valued slice bounds caused.
        valid = torch.nonzero(defe_mask)  # [N, 3] = (b, i, j)
        counts = torch.bincount(valid[:, 0], minlength=B).tolist()
        for b in range(B):
            if counts[b] == 0:
                # open a file to save these prints
                with open("no_valid_windows.log", "a") as f:
                    f.write(f"batch {b} has no valid windows\n")
                    f.write(f"defe_feature_filtered_nonzero: {torch.nonzero(defe_feature_filtered[b])}\n")
                    f.write(f"defe_feature_filtered: {defe_feature_filtered[b].size()}\n")
                    f.write(f"defe_mask: {defe_mask[b].size()}\n")
            assert counts[b] > 0, "No valid windows found"

        b_idx, i_idx, j_idx = valid.unbind(1)

        # 批量处理窗口特征
        window_features = windows[b_idx, i_idx, j_idx]  # [N, C, ws, ws]
        glob_windows = glob_pos_embed.reshape(C, num_win_h, window_size, num_win_w, window_size).permute(1, 3, 0, 2, 4)
        glob_pos_embeds = glob_windows[i_idx, j_idx]  # [N, C, ws, ws]
        across_mask = b_idx[:, None] != b_idx[None, :]

        encoded_features = self._encode_features(window_features, rel_pos_embed, glob_pos_embeds, across_mask)

        # # 特征重建: add each window's encoding back onto its patch of the map
        delta = encoded_features.new_zeros((B, num_win_h, num_win_w, C, window_size, window_size))
        delta = delta.index_put((b_idx, i_idx, j_idx), encoded_features)
        delta = delta.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        # computed in the wider dtype and rounded once, like the in-place `+=` was
        reconstructed = (backbone_memory + delta).to(backbone_memory.dtype)

        return reconstructed, defe_mask

    def _prepare_windows(self, features, mask, window_size):
        """预处理窗口划分和掩码（最大池化优化版）"""
        B, C, H_feat, W_feat = features.shape
        H_mask, W_mask = mask.shape[-2:]  # 原始mask的高分辨率尺寸

        num_win_h = H_feat // window_size
        num_win_w = W_feat // window_size

        # 计算池化核尺寸和步长
        kernel_h = H_mask // H_feat * window_size
        kernel_w = W_mask // W_feat * window_size
        stride_h = kernel_h
        stride_w = kernel_w

        # 划分特征图窗口 [B, num_win_h, num_win_w, C, window_size, window_size]
        windows = features.view(B, C, num_win_h, window_size, num_win_w, window_size).permute(0, 2, 4, 1, 3, 5)

        # 最大池化统计窗口有效性
        mask_float = mask.float()  # 转换为浮点型以支持池化
        pooled_mask = F.max_pool2d(mask_float, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))
        defe_mask = pooled_mask.squeeze(1) > 0  # [B, num_win_h, num_win_w]

        return windows, defe_mask

    def _encode_features(self, features, rel_pos_embed, glob_pos_embeds, across_mask=None):
        """三层编码处理"""
        # 调整维度 [N, C+D, h, w] -> [N, L, C+D]
        B, C, h, w = features.shape
        features = features.reshape(B, C, -1).permute(0, 2, 1)

        features = self.window_encoder(
            features, pos_embed=rel_pos_embed, glob_pos_embeds=glob_pos_embeds, across_mask=across_mask
        )

        # 恢复空间维度
        return features.permute(0, 2, 1).view(B, C, h, w)

    def _get_rel_embedding(self, window_size):
        """相对位置编码"""
        h, w = window_size
        weight = self.rel_pos_encoder[0].weight
        # the coordinate grid is a constant: upload it once, not every forward
        key = (h, w, weight.device, weight.dtype)
        coords = self._coords_cache.get(key)
        if coords is None:
            coords = self._coords_cache[key] = self._get_relative_coords(h, w).to(weight.device, weight.dtype)
        return self.rel_pos_encoder(coords).permute(2, 0, 1)

    @staticmethod
    def _get_relative_coords(h, w):
        """生成归一化坐标矩阵"""
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        return torch.stack([grid_x / (w - 1), grid_y / (h - 1)], dim=-1)
