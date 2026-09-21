"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Optional

import numpy as np
import PIL
import PIL.Image
import torch
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import SanitizeBoundingBoxes
from torchvision.tv_tensors import BoundingBoxes, Image, Mask, Video

from ...core import register
from .._misc import _boxes_keys, convert_to_tv_tensor


torchvision.disable_beta_transforms_warning()

RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
RandomVerticalFlip = register()(T.RandomVerticalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[list[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale,
            max_scale,
            min_aspect_ratio,
            max_aspect_ratio,
            sampler_options,
            trials,
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


# --------------------------------------------------------------------------- #
# Overhead imagery at a fixed ground sample distance (the small-boat set: ~1 m/px,
# 1024 px tiles, boats of 4-12 px). What matters here is different from COCO:
# apparent size carries information and must be preserved, orientation carries
# none, and positives are scarce (~1.9 boxes per tile).
# --------------------------------------------------------------------------- #
def _unpack(inputs):
    """``Compose`` hands transforms ``(image, target, dataset)``; keep the tail."""
    sample = inputs if len(inputs) > 1 else inputs[0]
    return sample[0], sample[1], tuple(sample[2:])


@register()
class RandomRotation90(T.Transform):
    """Rotate the tile by 90, 180 or 270 degrees.

    Nadir imagery has no canonical "up" -- a boat heading north is the same
    object as one heading east -- so this and the two flips together give the
    full 8-fold dihedral group, roughly 8x the effective data. Multiples of 90
    degrees resample nothing, which matters when the object is 6 px across.

    The default ``p`` of 0.75 spreads the four rotations evenly (no rotation
    keeps the remaining 25%).
    """

    def __init__(self, p: float = 0.75) -> None:
        super().__init__()
        self.p = p

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if torch.rand(1) >= self.p:
            return sample

        image, target, rest = _unpack(inputs)
        angle = 90.0 * int(torch.randint(1, 4, (1,)).item())
        image = F.rotate(image, angle, expand=True)
        target = dict(target)
        for key in ("boxes", "masks"):
            if key in target:
                target[key] = F.rotate(target[key], angle, expand=True)
        return (image, target, *rest)


@register()
class RandomZoomCrop(T.Transform):
    """Crop a window and blow it back up to the tile size, i.e. zoom *in* only.

    Ground sample distance is fixed for this project, so the apparent size of a
    boat is a feature rather than a nuisance: ``RandomZoomOut`` (which shrinks
    the tile by up to 4x by default) pushes a 6 px boat below what even a
    stride-4 feature map can hold. Magnifying is the safe direction -- it is
    what a finer-GSD scene looks like -- and the shifting crop window doubles as
    translation jitter.

    With a median of one boat per tile a blind crop often lands on empty water,
    so ``keep_object`` of the crops are placed around a boat that is present.
    """

    def __init__(
        self,
        min_scale: float = 1.0,
        max_scale: float = 1.5,
        keep_object: float = 0.9,
        p: float = 0.8,
    ) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.keep_object = keep_object
        self.p = p

    def _origin(self, extent, window, center):
        """Window start along one axis, containing ``center`` when given."""
        if window >= extent:
            return 0
        if center is None:
            return int(torch.randint(0, extent - window + 1, (1,)).item())
        low = max(0, int(center) - window + 1)
        high = min(extent - window, int(center))
        if high <= low:
            return max(0, min(low, extent - window))
        return int(torch.randint(low, high + 1, (1,)).item())

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if torch.rand(1) >= self.p:
            return sample

        image, target, rest = _unpack(inputs)
        height, width = F.get_size(image)
        scale = float(torch.empty(1).uniform_(self.min_scale, self.max_scale).item())
        window_h = max(1, min(height, int(round(height / scale))))
        window_w = max(1, min(width, int(round(width / scale))))
        if window_h == height and window_w == width:
            return sample

        center_y = center_x = None
        boxes = target.get("boxes")
        if boxes is not None and len(boxes) and torch.rand(1) < self.keep_object:
            box = boxes[int(torch.randint(0, len(boxes), (1,)).item())]
            center_x = float((box[0] + box[2]) / 2)
            center_y = float((box[1] + box[3]) / 2)

        top = self._origin(height, window_h, center_y)
        left = self._origin(width, window_w, center_x)

        crop = dict(top=top, left=left, height=window_h, width=window_w, size=[height, width])
        image = F.resized_crop(image, antialias=True, **crop)
        target = dict(target)
        for key in ("boxes", "masks"):
            if key in target:
                target[key] = F.resized_crop(target[key], antialias=True, **crop)
        return (image, target, *rest)


@register()
class CopyPasteSmallObjects(T.Transform):
    """Duplicate the boats already in the tile onto empty water elsewhere in it.

    The set averages ~1.9 boats per 1024 px tile, so a training step sees very
    few positives; copying a boat that is already there multiplies them without
    inventing appearances the sensor never produced. A paste site has to match
    the source patch's own surroundings in brightness and texture, which in
    practice means open water -- boats do not get stamped onto land, wakes or
    cloud. Edges are feathered so the network cannot key on a seam.

    Pastes come from the same tile on purpose: same sensor, sun angle and sea
    state as the original.
    """

    def __init__(
        self,
        p: float = 0.5,
        max_paste: int = 3,
        max_object_size: int = 32,
        margin: int = 3,
        bg_tolerance: float = 1.5,
        max_texture_ratio: float = 2.0,
        min_gap: int = 6,
        trials: int = 30,
    ) -> None:
        super().__init__()
        self.p = p
        self.max_paste = max_paste
        self.max_object_size = max_object_size
        self.margin = margin
        self.bg_tolerance = bg_tolerance
        self.max_texture_ratio = max_texture_ratio
        self.min_gap = min_gap
        self.trials = trials

    def _feather(self, height, width):
        """1 inside, ramping to 0 over ``margin`` px at the border."""
        ay, ax = np.ones(height, dtype=np.float32), np.ones(width, dtype=np.float32)
        margin = min(self.margin, height // 2, width // 2)
        if margin > 0:
            ramp = (np.arange(margin, dtype=np.float32) + 1) / (margin + 1)
            ay[:margin], ay[-margin:] = ramp, ramp[::-1]
            ax[:margin], ax[-margin:] = ramp, ramp[::-1]
        return np.minimum(ay[:, None], ax[None, :])

    def _free(self, candidate, occupied):
        """No overlap with anything already in the tile, plus a small gap."""
        x1, y1, x2, y2 = candidate
        for ox1, oy1, ox2, oy2 in occupied:
            if x1 < ox2 + self.min_gap and ox1 - self.min_gap < x2:
                if y1 < oy2 + self.min_gap and oy1 - self.min_gap < y2:
                    return False
        return True

    def forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if torch.rand(1) >= self.p:
            return sample

        image, target, rest = _unpack(inputs)
        boxes = target.get("boxes")
        # Only worth doing on the raw tile, before the image becomes a tensor.
        if not isinstance(image, PIL.Image.Image) or boxes is None or not len(boxes):
            return sample

        array = np.array(image)
        canvas_h, canvas_w = array.shape[:2]
        xyxy = boxes.detach().cpu().numpy()
        longest = np.maximum(xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1])
        sources = np.flatnonzero(longest <= self.max_object_size)
        if sources.size == 0:
            return sample

        occupied = [tuple(box) for box in xyxy]
        pasted, pasted_from = [], []
        for _ in range(int(torch.randint(1, self.max_paste + 1, (1,)).item())):
            index = int(sources[int(torch.randint(0, sources.size, (1,)).item())])
            x1 = max(0, int(np.floor(xyxy[index, 0])) - self.margin)
            y1 = max(0, int(np.floor(xyxy[index, 1])) - self.margin)
            x2 = min(canvas_w, int(np.ceil(xyxy[index, 2])) + self.margin)
            y2 = min(canvas_h, int(np.ceil(xyxy[index, 3])) + self.margin)
            patch = array[y1:y2, x1:x2]
            patch_h, patch_w = patch.shape[:2]
            if patch_h < 3 or patch_w < 3 or patch_h >= canvas_h or patch_w >= canvas_w:
                continue

            # The patch border is this boat's own background -- the water we
            # need to find again somewhere else in the tile.
            ring = np.concatenate(
                [patch[0].reshape(-1), patch[-1].reshape(-1), patch[:, 0].reshape(-1), patch[:, -1].reshape(-1)]
            ).astype(np.float32)
            ring_mean, ring_std = float(ring.mean()), float(ring.std())
            alpha = self._feather(patch_h, patch_w)
            if array.ndim == 3:
                alpha = alpha[..., None]

            for _ in range(self.trials):
                top = int(torch.randint(0, canvas_h - patch_h + 1, (1,)).item())
                left = int(torch.randint(0, canvas_w - patch_w + 1, (1,)).item())
                box = (
                    left + self.margin,
                    top + self.margin,
                    left + patch_w - self.margin,
                    top + patch_h - self.margin,
                )
                if box[2] - box[0] < 1 or box[3] - box[1] < 1 or not self._free(box, occupied):
                    continue
                region = array[top : top + patch_h, left : left + patch_w]
                target_stats = region.astype(np.float32)
                if abs(float(target_stats.mean()) - ring_mean) > self.bg_tolerance * (ring_std + 1.0):
                    continue
                if float(target_stats.std()) > self.max_texture_ratio * ring_std + 2.0:
                    continue

                blended = alpha * patch.astype(np.float32) + (1.0 - alpha) * target_stats
                array[top : top + patch_h, left : left + patch_w] = blended.astype(array.dtype)
                occupied.append(box)
                pasted.append(box)
                pasted_from.append(index)
                break

        if not pasted:
            return sample

        extra = torch.as_tensor(pasted, dtype=boxes.dtype)
        merged = torch.cat([boxes.as_subclass(torch.Tensor), extra], dim=0)
        target = dict(target)
        target["boxes"] = convert_to_tv_tensor(
            merged,
            key="boxes",
            box_format=boxes.format.value,
            spatial_size=getattr(boxes, _boxes_keys[1]),
        )
        source_index = torch.as_tensor(pasted_from, dtype=torch.long)
        for key in ("labels", "iscrowd"):
            if key in target and len(target[key]) == len(xyxy):
                target[key] = torch.cat([target[key], target[key][source_index]], dim=0)
        if "area" in target and len(target["area"]) == len(xyxy):
            areas = torch.as_tensor([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in pasted], dtype=target["area"].dtype)
            target["area"] = torch.cat([target["area"], areas], dim=0)

        return (PIL.Image.fromarray(array), target, *rest)


@register()
class RandomColorJitter(T.ColorJitter):
    """Illumination, sea-colour and haze jitter -- what actually differs between scenes.

    ``RandomPhotometricDistort``, the COCO default, also permutes the RGB
    channels, which turns the sea red or green. No sensor in this project does
    that, so it only spends capacity on a variation that will never be seen.
    """

    def __init__(
        self,
        brightness=(0.8, 1.25),
        contrast=(0.7, 1.4),
        saturation=(0.85, 1.15),
        hue=(-0.02, 0.02),
        p: float = 0.5,
    ) -> None:
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


@register()
class RandomGaussianBlur(T.GaussianBlur):
    """Mild defocus/haze jitter -- scenes differ in atmosphere and resampling.

    Kept gentle on purpose: a large sigma would simply erase a 6 px boat.
    """

    def __init__(self, kernel_size=3, sigma=(0.3, 0.8), p: float = 0.2) -> None:
        super().__init__(kernel_size, sigma)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt,
                key="boxes",
                box_format=self.fmt.upper(),
                spatial_size=spatial_size,
            )

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt
