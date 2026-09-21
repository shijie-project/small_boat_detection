"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import numpy as np
import torch

from src.zoo.dome.box_ops import box_iou


def dynamic_nms(boxes, scores, classes, iou_thresholds):
    unique_classes = classes.unique()
    keep_mask = torch.zeros_like(classes, dtype=torch.bool)
    for cls in unique_classes:
        cls_mask = classes == cls
        boxes_cls = boxes[cls_mask]
        scores_cls = scores[cls_mask]
        thresholds_cls = iou_thresholds[cls_mask]
        keep_cls = _per_class_dynamic_nms(boxes_cls, scores_cls, thresholds_cls)
        cls_indices = torch.nonzero(cls_mask, as_tuple=True)[0]
        keep_mask[cls_indices[keep_cls]] = True
    return torch.nonzero(keep_mask, as_tuple=True)[0]


def _per_class_dynamic_nms(boxes, scores, iou_thresholds):
    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size(0) == 1:
            break
        ious, _ = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])
        ious = ious.squeeze(0)
        suppress = ious >= iou_thresholds[i]
        idxs = idxs[1:][~suppress]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def dynamic_nms_indices(boxes, scores, classes, iou_thresholds):
    """Class-wise greedy NMS where each kept box suppresses with its own threshold.

    Returns the kept indices (ascending) as a numpy array. Same decisions as
    ``dynamic_nms_fast_legacy``, but the suppression matrix of all classes is
    computed on the device in one go and copied to the host once, where the
    inherently sequential greedy walk is cheap; the old loop synchronised on
    every candidate box (`if not keep_flags[i]` on a CUDA tensor).
    """
    num = boxes.shape[0]
    if num == 0:
        return np.zeros(0, dtype=np.int64)
    # Highest score first; a box can only be suppressed by an earlier one of
    # its own class, so walking all classes in one score order changes nothing.
    order = scores.argsort(descending=True, stable=True)
    boxes_sorted = boxes[order]
    iou_matrix, _ = box_iou(boxes_sorted, boxes_sorted)
    classes_sorted = classes[order]
    suppress = (iou_matrix >= iou_thresholds[order][:, None]) & (classes_sorted[:, None] == classes_sorted[None, :])

    suppress = suppress.cpu().numpy()
    removed = np.zeros(num, dtype=bool)
    keep = []
    for i in range(num):
        if removed[i]:
            continue
        keep.append(i)
        removed[i + 1 :] |= suppress[i, i + 1 :]
    return np.sort(order.cpu().numpy()[keep])


def dynamic_nms_fast(boxes, scores, classes, iou_thresholds):
    keep = dynamic_nms_indices(boxes, scores, classes, iou_thresholds)
    return torch.as_tensor(keep, dtype=torch.int64).to(boxes.device)


def dynamic_nms_fast_legacy(boxes, scores, classes, iou_thresholds):
    unique_classes = classes.unique()
    keep_mask = torch.zeros_like(classes, dtype=torch.bool)
    for cls in unique_classes:
        cls_mask = classes == cls
        boxes_cls = boxes[cls_mask]
        scores_cls = scores[cls_mask]
        thresholds_cls = iou_thresholds[cls_mask]
        keep_cls = _per_class_dynamic_nms_vectorized(boxes_cls, scores_cls, thresholds_cls)
        cls_indices = torch.nonzero(cls_mask, as_tuple=True)[0]
        keep_mask[cls_indices[keep_cls]] = True
    return torch.nonzero(keep_mask, as_tuple=True)[0]


def _per_class_dynamic_nms_vectorized(boxes, scores, iou_thresholds):
    # 按分数降序排列
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    thresholds = iou_thresholds[order]

    # 预计算所有框之间的 IoU 矩阵（对称矩阵）
    iou_matrix, _ = box_iou(boxes, boxes)

    num = boxes.shape[0]
    keep_flags = torch.ones(num, dtype=torch.bool, device=boxes.device)
    keep = []
    for i in range(num):
        if not keep_flags[i]:
            continue
        keep.append(i)
        # 对于排序后位于 i 后面的所有框，如果 IoU 大于等于当前框的动态阈值，则置为 False
        # 注意：这里仅比较后面的框，避免重复判断
        if i < num - 1:
            # mask 指示第 i 框与后续各框的 IoU 是否超过阈值 thresholds[i]
            mask = iou_matrix[i, (i + 1) :] >= thresholds[i]
            keep_flags[(i + 1) :] &= ~mask
    # 返回在原始排序中的索引，再映射回原始索引
    return order[torch.tensor(keep, device=boxes.device)]
