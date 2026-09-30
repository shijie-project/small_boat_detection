"""
A detection dataset kept on disk as a COCO file and a folder of images: what annotating in Label
Studio produces (``tools/annotation/ls_to_coco.py``), the AEA small-boat tiles among them.

It is an ``HFDetection`` in every respect but where the rows come from: the rows are the COCO
file's images with their annotations, read once, and the pixels are opened from ``img_folder``.
Targets, the COCO ground truth the evaluators score against (image ids are row indices, category
ids the labels) and the pickling for the loader's workers are the base class's, so a local dataset
and a Hub one cannot differ in how they are trained or scored.
"""

import json
import os
from collections import defaultdict
from collections.abc import Callable

import torch
from PIL import Image

from ...core import register
from .hf_detection import HFDetection

__all__ = ["CocoDetection"]


@register()
class CocoDetection(HFDetection):
    """
    ``ann_file`` (a COCO json) over the images in ``img_folder``, matched by ``file_name``. Every
    image the file lists is kept, those without a box too: on sparse data such as AEA (a tile
    holds about two boats, and one in seven holds none) the empty tiles are the open water the
    model has to learn to leave alone.

    ``classes`` names the categories to train on, by the file's category names: their boxes are
    read and relabelled 0..K-1 in the order given, so ``num_classes`` is K, and every other box
    is background. ``None`` keeps every category with its id from the file, so ``num_classes``
    has to exceed the largest id. Boxes with ``iscrowd`` are ignore regions, as in the base class.
    """

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        classes: list[str] | None = None,
        transforms: Callable | None = None,
    ):
        self.img_folder = os.path.expanduser(img_folder)
        self.ann_file = os.path.expanduser(ann_file)
        self.classes = classes
        self.transforms = transforms
        self._coco = None
        self._hf_meta = None

        with open(self.ann_file, encoding="utf-8") as f:
            data = json.load(f)
        names = {c["id"]: c["name"] for c in data.get("categories", [])}
        if classes is None:
            self._label_map = None
            self.CATEGORIES = sorted(names.items())
        else:
            unknown = [c for c in classes if c not in names.values()]
            if unknown:
                raise ValueError(f"{self.ann_file} has no categories {unknown}; available: {sorted(names.values())}")
            self._label_map = {cid: classes.index(name) for cid, name in names.items() if name in classes}
            self.CATEGORIES = list(enumerate(classes))

        objects = defaultdict(lambda: {"bbox": [], "category": [], "iscrowd": []})
        for ann in data.get("annotations", []):
            entry = objects[ann["image_id"]]
            entry["bbox"].append(ann["bbox"])
            entry["category"].append(ann["category_id"])
            entry["iscrowd"].append(ann.get("iscrowd", 0))
        # the rows HFDetection reads, without the pixels: its hf_meta
        images = sorted(data["images"], key=lambda image: image["id"])
        self.hf = [
            {
                "id": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "objects": objects[image["id"]],
            }
            for image in images
        ]
        # the file's own ids, for detections that leave the process (predictions.json): the ground
        # truth the evaluators score against has row indices for image ids and labels for categories
        self.source_image_ids = [image["id"] for image in images]
        self.source_category_ids = (
            {label: label for label, _ in self.CATEGORIES}
            if self._label_map is None
            else {label: cid for cid, label in self._label_map.items()}
        )

    @property
    def hf_meta(self):
        return self.hf

    def parse_objects(self, objects):
        boxes = torch.tensor(objects["bbox"], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # COCO xywh -> xyxy
        category = torch.tensor(objects["category"], dtype=torch.int64)
        iscrowd = torch.tensor(objects["iscrowd"], dtype=torch.int64)
        if self._label_map is not None:
            category = torch.tensor([self._label_map.get(c, -1) for c in category.tolist()], dtype=torch.int64)
            keep = category >= 0
            boxes, category, iscrowd = boxes[keep], category[keep], iscrowd[keep]
        return boxes, category, iscrowd

    def file_name(self, row_id):
        return row_id

    def load_item(self, index: int):
        row = self.hf[index]
        image = Image.open(os.path.join(self.img_folder, row["id"]))
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        boxes, labels, iscrowd = self.parse_objects(row["objects"])
        return image, self._make_target(index, boxes, labels, iscrowd, w, h)

    def __repr__(self):
        s = f"{self.__class__.__name__}(img_folder={self.img_folder!r}, ann_file={self.ann_file!r}, len={len(self)})"
        if self.transforms is not None:
            s += f"\n  transforms:\n    {self.transforms!r}"
        return s
