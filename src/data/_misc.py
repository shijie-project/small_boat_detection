"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import importlib.metadata

import torch
from torch import Tensor


if "0.15.2" in importlib.metadata.version("torchvision"):
    import torchvision

    torchvision.disable_beta_transforms_warning()

    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat, Mask

    _boxes_keys = ["format", "spatial_size"]

elif "0.17" > importlib.metadata.version("torchvision") >= "0.16":
    import torchvision
    from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask

    torchvision.disable_beta_transforms_warning()

    _boxes_keys = ["format", "canvas_size"]

elif importlib.metadata.version("torchvision") >= "0.17":
    from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask

    _boxes_keys = ["format", "canvas_size"]

else:
    raise RuntimeError("Please make sure torchvision version >= 0.15.2")


class SharedEpoch:
    """An epoch counter that DataLoader worker processes see change.

    Persistent workers keep the dataset / collate_fn copy they started with, so a
    plain attribute set in the main process afterwards never reaches them; a
    shared-memory tensor does, under both fork and spawn.
    """

    def __init__(self, epoch: int = -1) -> None:
        self._value = torch.full((1,), epoch, dtype=torch.int64).share_memory_()

    def set(self, epoch: int) -> None:
        self._value[0] = epoch

    def get(self) -> int:
        return int(self._value[0])


class EpochMixin:
    """``set_epoch`` / ``epoch`` backed by a :class:`SharedEpoch` once one exists."""

    def init_shared_epoch(self) -> None:
        if "_shared_epoch" not in self.__dict__:
            self._shared_epoch = SharedEpoch(self._epoch if hasattr(self, "_epoch") else -1)

    def set_epoch(self, epoch) -> None:
        self._epoch = epoch
        if "_shared_epoch" in self.__dict__:
            self._shared_epoch.set(epoch)

    @property
    def epoch(self):
        if "_shared_epoch" in self.__dict__:
            return self._shared_epoch.get()
        return self._epoch if hasattr(self, "_epoch") else -1


def convert_to_tv_tensor(tensor: Tensor, key: str, box_format="xyxy", spatial_size=None) -> Tensor:
    """
    Args:
        tensor (Tensor): input tensor
        key (str): transform to key

    Return:
        Dict[str, TV_Tensor]
    """
    assert key in ("boxesmasks"), "Only support 'boxes' and 'masks'"

    if key == "boxes":
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == "masks":
        return Mask(tensor)
