"""
Tiled inference for VERY large satellite images, based on torch_inf.py.

Strategy
--------
A full satellite image (e.g. ~11000 x 10500) is far too big to feed to the
detector at once. We slide a window over it:

  * crop a TILE x TILE region of real content (default 1024, i.e. the full
    network input size) -- this keeps every pixel at its ORIGINAL resolution
    (no resize of the content) so small boats are not shrunk
  * paste it top-left into a PAD x PAD canvas (default 1024). For interior
    tiles the crop already fills the whole canvas; only right/bottom edge tiles
    (whose content is smaller than TILE) get zero-padded to reach PAD.
  * to avoid cutting a boat at a tile border, windows OVERLAP (default 24 px),
    i.e. stride = TILE - overlap = 1000. Two consecutive tiles cover
    1000 + (24 overlap) + 1000. A boat split at one tile's edge appears whole
    in the neighbouring tile, and duplicate detections in the overlap are
    removed by a final global NMS.

Each tile is run exactly like torch_inf.py (resize PAD -> INPUT_SIZE, which is
a no-op at the default 1024, and the postprocessor rescales boxes to the 1024
canvas), boxes are filtered to the real content area, shifted by the tile
origin into full-image coordinates, gathered across all tiles, and
de-duplicated with class-wise NMS.

Usage
-----
python torch_inf_tiled.py \
    -c <config.yml> -r <checkpoint.pth> \
    -i big_image.png -d cuda \
    --tile 1024 --pad 1024 --overlap 24 --thrh 0.4 --batch 8

Outputs (next to the input, same basename):
  *_det.jpg   full-resolution image with boxes drawn
  *_det.json  detections [{bbox_xyxy, score, label}, ...] in full-image coords
"""

import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torchvision.ops import batched_nms


# huge satellite images exceed PIL's default decompression-bomb guard
Image.MAX_IMAGE_PIXELS = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


def build_model(args):
    """Load the model exactly like torch_inf.py:main()."""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if not args.resume:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(args.device).eval()
    return model


def gen_windows(W, H, tile, overlap):
    """Yield (x0, y0, w, h) content windows covering the WxH image.

    Stride = tile - overlap. Edge windows are clamped so the last column/row
    still reaches the image border (content may be smaller than `tile` there).
    """
    stride = tile - overlap
    assert stride > 0, "overlap must be smaller than tile"

    xs = list(range(0, max(W - tile, 0) + 1, stride))
    ys = list(range(0, max(H - tile, 0) + 1, stride))
    if not xs or xs[-1] != max(W - tile, 0):
        xs.append(max(W - tile, 0))
    if not ys or ys[-1] != max(H - tile, 0):
        ys.append(max(H - tile, 0))

    for y0 in ys:
        for x0 in xs:
            w = min(tile, W - x0)
            h = min(tile, H - y0)
            yield x0, y0, w, h


@torch.no_grad()
def run_tiles(model, device, img, args, crops_dir=None):
    """Run the detector over all tiles. Returns global boxes/scores/labels.

    If `crops_dir` is given, every 1024x1024 tile canvas is saved there with
    its own detection boxes drawn (before they are shifted into full-image
    coordinates), so each crop's raw result can be inspected on its own.
    """
    W, H = img.size
    to_tensor = T.Compose([T.Resize((args.input_size, args.input_size)), T.ToTensor()])

    windows = list(gen_windows(W, H, args.tile, args.overlap))
    print(f"Image {W}x{H} -> {len(windows)} tiles (tile={args.tile}, pad={args.pad}, overlap={args.overlap})")

    all_boxes, all_scores, all_labels = [], [], []
    batch_tensors, batch_meta = [], []
    # the postprocessor rescales to the FULL pad canvas size
    orig_size = torch.tensor([[args.pad, args.pad]], device=device)

    def flush():
        if not batch_tensors:
            return
        ims = torch.cat(batch_tensors, 0)
        sizes = orig_size.repeat(len(batch_tensors), 1)
        labels, boxes, scores = model(ims, sizes)
        for bi, (idx, x0, y0, cw, ch, canvas) in enumerate(batch_meta):
            b, s, l = boxes[bi], scores[bi], labels[bi]
            keep = s > args.thrh
            b, s, l = b[keep], s[keep], l[keep]
            # keep only the requested class(es), drop everything else
            if args.classes is not None and b.numel():
                cls_keep = torch.zeros_like(l, dtype=torch.bool)
                for c in args.classes:
                    cls_keep |= l == c
                b, s, l = b[cls_keep], s[cls_keep], l[cls_keep]
            # discard boxes whose center falls in the zero-padded region
            if b.numel():
                cx = (b[:, 0] + b[:, 2]) / 2
                cy = (b[:, 1] + b[:, 3]) / 2
                inside = (cx < cw) & (cy < ch)
                b, s, l = b[inside], s[inside], l[inside]
            # clamp to content box (boxes still in tile-local coords)
            if b.numel():
                b[:, [0, 2]] = b[:, [0, 2]].clamp(0, cw)
                b[:, [1, 3]] = b[:, [1, 3]].clamp(0, ch)

            # save this crop with its own (tile-local) boxes drawn
            if crops_dir is not None:
                tile_img = canvas.copy()
                d = ImageDraw.Draw(tile_img)
                for bb, ss, ll in zip(b.tolist(), s.tolist(), l.tolist()):
                    d.rectangle(bb, outline="red", width=3)
                    d.text((bb[0], bb[1]), f"{int(ll)} {ss:.2f}", fill="yellow")
                name = f"tile{idx:04d}_x{x0}_y{y0}_n{len(b)}.jpg"
                tile_img.save(os.path.join(crops_dir, name), quality=90)

            if b.numel() == 0:
                continue
            # shift into full-image coordinates
            b = b.clone()
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            all_boxes.append(b.cpu())
            all_scores.append(s.cpu())
            all_labels.append(l.cpu())
        batch_tensors.clear()
        batch_meta.clear()

    for i, (x0, y0, cw, ch) in enumerate(windows):
        crop = img.crop((x0, y0, x0 + cw, y0 + ch))
        # paste content top-left into a PAD x PAD black canvas (no content resize)
        canvas = Image.new("RGB", (args.pad, args.pad), (0, 0, 0))
        canvas.paste(crop, (0, 0))
        batch_tensors.append(to_tensor(canvas).unsqueeze(0).to(device))
        batch_meta.append((i, x0, y0, cw, ch, canvas if crops_dir is not None else None))
        if len(batch_tensors) >= args.batch:
            flush()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(windows)} tiles queued")
    flush()

    if not all_boxes:
        return (torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64))

    boxes = torch.cat(all_boxes, 0)
    scores = torch.cat(all_scores, 0)
    labels = torch.cat(all_labels, 0)

    # global class-wise NMS removes duplicates from overlapping tiles
    keep = batched_nms(boxes, scores, labels, args.nms_iou)
    print(f"Detections: {len(boxes)} -> {len(keep)} after global NMS")
    return boxes[keep], scores[keep], labels[keep]


def save_results(img, boxes, scores, labels, out_base):
    import json

    draw = ImageDraw.Draw(img)
    dets = []
    for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        draw.rectangle(b, outline="red", width=3)
        draw.text((b[0], b[1]), f"{int(l)} {s:.2f}", fill="yellow")
        dets.append({"bbox_xyxy": [round(v, 1) for v in b], "score": round(s, 4), "label": int(l)})

    jpg = out_base + "_det.jpg"
    js = out_base + "_det.json"
    img.save(jpg, quality=90)
    with open(js, "w") as f:
        json.dump(dets, f)
    print(f"Saved {jpg}\nSaved {js}  ({len(dets)} boxes)")


def main(args):
    model = build_model(args)
    img = Image.open(args.input).convert("RGB")

    # one output folder per image: <input_dir>/<basename>_det/
    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)), stem + "_det")
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    boxes, scores, labels = run_tiles(model, args.device, img, args, crops_dir=crops_dir)
    save_results(img, boxes, scores, labels, os.path.join(out_dir, stem))
    print(f"Per-crop images in {crops_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=str, required=True)
    p.add_argument("-r", "--resume", type=str, required=True)
    p.add_argument("-i", "--input", type=str, required=True)
    p.add_argument("-d", "--device", type=str, default="cuda")
    p.add_argument("--tile", type=int, default=1024, help="content crop size")
    p.add_argument("--pad", type=int, default=1024, help="padded canvas size (postprocessor rescales boxes to this)")
    p.add_argument(
        "--input_size",
        type=int,
        default=1024,
        help="size actually fed to the net; MUST match the config's eval_spatial_size "
        "(e.g. 800 for Dome-M-AITOD). Set --tile --pad --input_size all equal to keep "
        "content at original resolution with no resize.",
    )
    p.add_argument("--overlap", type=int, default=24, help="overlap between tiles (px); stride = tile - overlap")
    p.add_argument("--thrh", type=float, default=0.4, help="score threshold")
    p.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[3],
        help="only keep these class label(s); default only class 3. Pass e.g. "
        "--classes 3 for one class, --classes 1 3 for several, or omit for all",
    )
    p.add_argument("--nms_iou", type=float, default=0.5, help="global NMS IoU")
    p.add_argument("--batch", type=int, default=8, help="tiles per forward pass")
    args = p.parse_args()
    main(args)
