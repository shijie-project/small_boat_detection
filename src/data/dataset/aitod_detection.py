"""
Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.

AI-TOD-v2, read from the Hub (https://huggingface.co/datasets/shijli/aitod-v2).
"""

from collections.abc import Callable

import torch

from ...core import register
from .hf_detection import HFDetection

__all__ = ["AITODDetection"]


# the official AI-TOD-v2 category ids, 0..7, by name
AITOD_CLASSES = [
    "airplane",
    "bridge",
    "storage-tank",
    "ship",
    "swimming-pool",
    "vehicle",
    "person",
    "wind-mill",
]


@register()
class AITODDetection(HFDetection):
    """
    The ``detection`` config of the AI-TOD-v2 repo: 11214 train / 2804 validation / 14018 test
    images of 800 x 800 pixels, with the official v2 annotations whole. Labels are the official
    category ids 0..7, all of them scored.

    The AI-TOD protocol trains on train + validation and reports on test; pass
    ``split: train+validation`` for that. There are no ignore regions: every box is an object,
    so the crowd flag is 0 everywhere. The 51 zero-width or zero-height boxes of the release are
    dropped from the training target by ``HFDetection`` and kept in the ground truth, where
    COCOeval scores them like any other box.
    """

    REPO = "shijli/aitod-v2"
    CONFIG = "detection"
    CATEGORIES = list(enumerate(AITOD_CLASSES))

    def __init__(
        self,
        split: str,
        classes: list[str] | None = None,
        repo: str | None = None,
        config: str | None = None,
        transforms: Callable | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
    ):
        """
        ``classes`` narrows the dataset to a subset of the categories, by name: only the images
        with at least one box of them are kept, only their boxes are read (every other object is
        background), and they are relabelled 0..K-1 in the order given, so ``num_classes`` is K.
        ``classes: [ship]`` is the ship-only AI-TOD. ``None`` keeps all eight with their ids.
        """
        super().__init__(split, repo, config, transforms, revision, cache_dir)
        self.classes = classes
        if classes is None:
            self._label_map = None
            return

        unknown = [c for c in classes if c not in AITOD_CLASSES]
        if unknown:
            raise ValueError(f"unknown AI-TOD classes {unknown}; available: {AITOD_CLASSES}")
        self._label_map = torch.full((len(AITOD_CLASSES),), -1, dtype=torch.int64)
        for new_id, name in enumerate(classes):
            self._label_map[AITOD_CLASSES.index(name)] = new_id
        self.CATEGORIES = list(enumerate(classes))

        wanted = {AITOD_CLASSES.index(c) for c in classes}
        categories = self.hf.select_columns(["objects"])["objects"]
        keep = [i for i, objects in enumerate(categories) if wanted.intersection(objects["category"])]
        self.hf = self.hf.select(keep)

    def parse_objects(self, objects):
        category = torch.tensor(objects["category"], dtype=torch.int64)
        boxes = torch.tensor(objects["bbox"], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # official xywh -> xyxy
        if self._label_map is not None:
            category = self._label_map[category]
            boxes, category = boxes[category >= 0], category[category >= 0]
        iscrowd = torch.zeros_like(category)
        return boxes, category, iscrowd

    def file_name(self, row_id):
        return row_id + ".png"
