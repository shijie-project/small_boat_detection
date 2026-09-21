"""Cut very large satellite images into tiles for annotation.

The simple scheme: pad the image up to a whole number of tiles (black on the
right and bottom edges) and cut it into a plain ``rows x cols`` grid. Every
pixel keeps its original resolution -- nothing is resized -- so a boat is the
same size in a tile as it is in the source image, which is what the detector
and the annotator both need.

Most of a satellite scene is empty water, so the whole grid is usually far more
tiles than anybody wants to annotate. ``--regions`` cuts only the rectangles
listed in a JSON file you write yourself; everything else about the cut --
resolution, padding, the manifest -- is unchanged.

Padding is never materialised: each tile is a black ``tile x tile`` canvas with
the real content pasted top-left, so only the edge tiles carry any padding and
a 500 MB source is never copied whole.

Alongside the tiles goes ``tiles.json``, which records the source size and the
origin of every tile. That is what maps a box drawn on a tile back onto the
full image later.

Usage
-----
    # every image in the folder -> ../data/satellite_images/split_images/<stem>/
    python tools/dataset/tile_satellite.py -i ../data/satellite_images

    # one image, custom grid and output
    python tools/dataset/tile_satellite.py -i big.png --tile 1024 --overlap 0 -o out/

    # skip tiles that are a single flat colour (all-black padding, no-data)
    python tools/dataset/tile_satellite.py -i ../data/satellite_images --skip-blank

    # drop the black background: tiles that are >= 99% black are not written
    python tools/dataset/tile_satellite.py -i ../data/satellite_images --drop-black

    # cut only the hand-drawn rectangles, gridded inside each one
    python tools/dataset/tile_satellite.py -i big.png --regions regions.json

    # ... or each rectangle whole, as one image of its own size
    python tools/dataset/tile_satellite.py -i big.png --regions regions.json --whole-regions
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

import cv2
import numpy as np


try:
    import tifffile
except ImportError:  # only needed for the SAR GeoTIFFs
    tifffile = None


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
# Output tiles live here, one folder per source image.
SPLIT_DIRNAME = "split_images"
# Folders we write ourselves, so a second run does not tile its own output.
SKIP_DIRS = (SPLIT_DIRNAME, "crops")


def to_uint8(arr, pmin=1.0, pmax=99.0):
    """8-bit view of an image, percentile-stretched if it is not 8-bit already.

    Same treatment as ``tif_to_png.py``: SAR amplitude (uint16/float, hugely
    skewed) turns into a near-black image under a plain cast, so the stretch is
    what makes the targets visible at all.
    """
    if arr.dtype == np.uint8:
        return arr

    a = arr.astype(np.float32)
    finite = a[np.isfinite(a)]
    valid = finite[finite > 0]  # SAR uses 0 as no-data
    sample = valid if valid.size else finite
    lo = np.percentile(sample, pmin)
    hi = np.percentile(sample, pmax)
    if hi <= lo:
        lo, hi = float(a.min()), float(max(a.max(), a.min() + 1))
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    return (a * 255.0 + 0.5).astype(np.uint8)


def load_image(path):
    """The whole image as 8-bit BGR, whatever it was stored as.

    OpenCV first: it reads the JPEG-compressed TIFFs that tifffile cannot
    without imagecodecs. tifffile picks up the uint16 GeoTIFFs OpenCV declines.
    """
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None and tifffile is not None:
        arr = tifffile.imread(path)  # raises with its own message if it cannot
    if arr is None:
        raise RuntimeError(
            f"could not read {path} -- for a SAR GeoTIFF, convert it with data/satellite_images/tif_to_png.py first"
        )

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    arr = to_uint8(arr)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr


def grid(width, height, tile, overlap):
    """``(cols, rows, stride)`` -- enough windows to cover the image once."""
    stride = tile - overlap
    if stride < 1:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile ({tile})")
    cols = max(1, math.ceil((width - tile) / stride) + 1)
    rows = max(1, math.ceil((height - tile) / stride) + 1)
    return cols, rows, stride


# --------------------------------------------------------------------------- #
# Windows: where the tiles are taken from, the whole image or marked regions.
# --------------------------------------------------------------------------- #
def grid_windows(width, height, tile, overlap):
    """Every window of the plain full-image grid, in reading order."""
    cols, rows, stride = grid(width, height, tile, overlap)
    return [
        {"region": None, "row": row, "col": col, "x": col * stride, "y": row * stride, "w": tile, "h": tile}
        for row in range(rows)
        for col in range(cols)
    ]


def region_windows(regions, tile, overlap, whole):
    """The windows inside the hand-drawn regions.

    ``whole`` takes each region as a single image of its own size; otherwise a
    region is gridded the same way the full image would be, anchored at its own
    top-left corner. A window on the right or bottom edge of a region is
    allowed to run past it -- real pixels beyond the line the hand drew beat a
    strip of black padding.
    """
    stride = tile - overlap
    if stride < 1:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile ({tile})")

    windows = []
    for index, (x, y, w, h) in enumerate(regions):
        if whole:
            windows.append({"region": index, "row": 0, "col": 0, "x": x, "y": y, "w": w, "h": h})
            continue
        cols = max(1, math.ceil((w - tile) / stride) + 1)
        rows = max(1, math.ceil((h - tile) / stride) + 1)
        windows += [
            {
                "region": index,
                "row": row,
                "col": col,
                "x": x + col * stride,
                "y": y + row * stride,
                "w": tile,
                "h": tile,
            }
            for row in range(rows)
            for col in range(cols)
        ]
    return windows


def as_box(entry):
    """One region as ``(x, y, w, h)``, from either shape the file may use."""
    if isinstance(entry, dict):
        values = [entry.get(key) for key in ("x", "y", "w", "h")]
    else:
        values = list(entry)[:4]
    if len(values) != 4 or any(value is None for value in values):
        raise SystemExit(f"bad region (need x, y, w, h): {entry!r}")
    return tuple(int(round(float(value))) for value in values)


def clip_regions(regions, width, height):
    """Regions cut back to what is actually inside the image; empties dropped."""
    kept = []
    for entry in regions:
        x, y, w, h = as_box(entry)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 - x0 > 0 and y1 - y0 > 0:
            kept.append((x0, y0, x1 - x0, y1 - y0))
    return kept


def load_regions(path):
    """``{stem: [region, ...]}`` from the regions file, or ``None`` if unused.

    Two shapes are accepted: ``{"regions": [...]}`` for a single image, and
    ``{"images": {stem: [...]}}`` for a whole folder marked up in one go. A
    bare list is read as the first.
    """
    if not path:
        return None
    with open(path) as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return {"": data}
    if isinstance(data, dict) and isinstance(data.get("images"), dict):
        return {stem: list(entries) for stem, entries in data["images"].items()}
    if isinstance(data, dict) and isinstance(data.get("regions"), list):
        return {"": data["regions"]}
    raise SystemExit(f"{path}: expected a list, {{'regions': [...]}} or {{'images': {{...}}}}")


def regions_for(table, stem):
    """This image's regions: its own entry, else the single-image one."""
    if stem in table:
        return table[stem]
    return table.get("", None) if len(table) == 1 and "" in table else None


def tile_name(stem, window, suffix):
    """``<stem>_g<region>_r<row>_c<col>`` -- the region part only when there is one."""
    part = "" if window["region"] is None else f"_g{window['region']:02d}"
    return f"{stem}{part}_r{window['row']:03d}_c{window['col']:03d}{suffix}"


def is_blank(tile_img):
    """A tile with nothing in it: one flat colour, padding included."""
    return bool(tile_img.max() == tile_img.min())


def black_ratio(tile_img, level):
    """How much of the tile is black.

    ``level`` is the slack: a "black" background that has been through JPEG
    lands on values of 1-5 rather than a clean 0, and a strict ``== 0`` test
    would keep every one of those tiles.
    """
    dark = tile_img.max(axis=2) if tile_img.ndim == 3 else tile_img
    return float((dark <= level).mean())


def drop_test(skip_blank, drop_black, black_frac, black_level):
    """``tile -> reason to throw it away``, or ``""`` to keep it.

    Black is checked first: an all-black tile is blank as well, and "black" is
    the answer that says which switch dropped it.
    """

    def reason(tile_img):
        if drop_black and black_ratio(tile_img, black_level) >= black_frac:
            return "black"
        if skip_blank and is_blank(tile_img):
            return "blank"
        return ""

    return reason


def write_tile(path, tile_img, quality):
    if path.lower().endswith((".jpg", ".jpeg")):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # fast; these are intermediates
    if not cv2.imwrite(path, tile_img, params):
        raise RuntimeError(f"failed to write {path}")


PROGRESS_EVERY = 25  # windows between progress lines, so a big grid still talks


def plan(width, height, tile, overlap, regions, whole):
    """``(windows, manifest fields)`` for one image, gridded or by region."""
    if regions is None:
        cols, rows, stride = grid(width, height, tile, overlap)
        padded = ((cols - 1) * stride + tile, (rows - 1) * stride + tile)
        windows = grid_windows(width, height, tile, overlap)
        print(
            f"  {width}x{height} -> {rows}x{cols} grid = {len(windows)} tiles "
            f"(tile={tile}, overlap={overlap}, padded to {padded[0]}x{padded[1]})"
        )
        fields = {
            "mode": "grid",
            "cols": cols,
            "rows": rows,
            "padded_width": padded[0],
            "padded_height": padded[1],
        }
        return windows, fields

    boxes = clip_regions(regions, width, height)
    windows = region_windows(boxes, tile, overlap, whole)
    how = "whole" if whole else f"tile={tile}, overlap={overlap}"
    print(f"  {width}x{height} -> {len(boxes)} region(s) = {len(windows)} tiles ({how})")
    for index, (x, y, w, h) in enumerate(boxes):
        count = sum(1 for window in windows if window["region"] == index)
        print(f"    region {index}: {w}x{h} at ({x}, {y}) -> {count} tiles")
    fields = {
        "mode": "regions",
        "regions": [{"x": x, "y": y, "w": w, "h": h} for x, y, w, h in boxes],
    }
    return windows, fields


def split_image(path, out_root, tile, overlap, suffix, quality, drop, regions=None, whole=False):
    """Cut one image into ``out_root/<stem>/``. Returns the number of tiles written."""
    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(out_root, stem)
    os.makedirs(out_dir, exist_ok=True)

    image = load_image(path)
    height, width = image.shape[:2]
    windows, fields = plan(width, height, tile, overlap, regions, whole)
    if not windows:
        print("  nothing to cut")
        return 0

    entries = []
    dropped = Counter()
    for done, window in enumerate(windows, 1):
        x0, y0 = window["x"], window["y"]
        box_w, box_h = window["w"], window["h"]
        content = image[y0 : y0 + box_h, x0 : x0 + box_w]
        content_h, content_w = content.shape[:2]

        if content_h == box_h and content_w == box_w:
            canvas = content
        else:  # right / bottom edge: pad out to a full tile
            canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
            canvas[:content_h, :content_w] = content

        why = drop(canvas)
        if why:
            dropped[why] += 1
        else:
            name = tile_name(stem, window, suffix)
            write_tile(os.path.join(out_dir, name), canvas, quality)
            entries.append(
                {
                    "name": name,
                    "region": window["region"],
                    "row": window["row"],
                    "col": window["col"],
                    "x": x0,
                    "y": y0,
                    "width": box_w,
                    "height": box_h,
                    "content_w": content_w,
                    "content_h": content_h,
                }
            )
        if done % PROGRESS_EVERY == 0 or done == len(windows):
            print(f"    {done}/{len(windows)} windows ({len(entries)} written)")

    manifest = {
        "source": os.path.abspath(path).replace("\\", "/"),
        "width": width,
        "height": height,
        "tile": tile,
        "overlap": overlap,
        "stride": tile - overlap,
        **fields,
        "skipped_blank": dropped["blank"],
        "skipped_black": dropped["black"],
        "tiles": entries,
    }
    with open(os.path.join(out_dir, "tiles.json"), "w") as handle:
        json.dump(manifest, handle, indent=1)

    counts = ", ".join(f"{count} {reason}" for reason, count in dropped.items() if count)
    skipped = f", {counts} skipped" if counts else ""
    print(f"  wrote {len(entries)} tiles{skipped} -> {out_dir}")
    return len(entries)


def collect(target, recursive):
    """The images to cut: one file, or the ones in a folder."""
    if os.path.isfile(target):
        return [target]
    if not os.path.isdir(target):
        raise SystemExit(f"no such file or directory: {target}")

    def keep(name):  # never descend into our own output, an inference run, or a cache
        return name not in SKIP_DIRS and "_det" not in name and not name.startswith(".")

    found = []
    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if keep(d)] if recursive else []
        found += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(IMAGE_SUFFIXES)]
    return found


def main(args):
    images = collect(args.input, args.recursive)
    if not images:
        raise SystemExit(f"no images ({', '.join(IMAGE_SUFFIXES)}) in {args.input}")

    suffix = "." + args.format.lower().lstrip(".")
    drop = drop_test(args.skip_blank, args.drop_black, args.black_frac, args.black_level)
    table = load_regions(args.regions)
    total = 0
    print(f"{len(images)} image(s) to split")
    if table is not None:
        print(f"cutting the regions in {args.regions}")
    if args.drop_black:
        print(f"dropping tiles >= {args.black_frac:.1%} black (pixels <= {args.black_level})")
    for index, path in enumerate(images, 1):
        print(f"[{index}/{len(images)}] {path}")
        regions = None
        if table is not None:
            regions = regions_for(table, os.path.splitext(os.path.basename(path))[0])
            if regions is None:
                print("  no regions marked for this image -- skipped")
                continue
        out_root = args.output or os.path.join(os.path.dirname(os.path.abspath(path)), SPLIT_DIRNAME)
        total += split_image(
            path,
            out_root,
            args.tile,
            args.overlap,
            suffix,
            args.quality,
            drop,
            regions=regions,
            whole=args.whole_regions,
        )
    print(f"done: {total} tiles from {len(images)} image(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="image file or folder of images")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=f"output root (default: <input folder>/{SPLIT_DIRNAME}); tiles go in <root>/<stem>/",
    )
    parser.add_argument("--tile", type=int, default=1024, help="tile size in px (default 1024)")
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="overlap between neighbouring tiles in px; stride = tile - overlap (default 0)",
    )
    parser.add_argument("--format", default="png", choices=["png", "jpg"], help="tile format")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality, --format jpg only")
    parser.add_argument(
        "--skip-blank",
        action="store_true",
        help="do not write tiles that are a single flat colour (all-black padding / no-data)",
    )
    parser.add_argument(
        "--drop-black",
        action="store_true",
        help="do not write tiles that are (almost) all black -- the no-data background",
    )
    parser.add_argument(
        "--black-frac",
        type=float,
        default=0.99,
        help="--drop-black: share of the tile that must be black to drop it (default 0.99)",
    )
    parser.add_argument(
        "--black-level",
        type=int,
        default=8,
        help="--drop-black: pixels this dark or darker count as black (default 8)",
    )
    parser.add_argument("--recursive", action="store_true", help="also descend into sub-folders of --input")
    parser.add_argument(
        "--regions",
        default=None,
        help="JSON of rectangles to cut instead of the full grid; images it does not mention are skipped",
    )
    parser.add_argument(
        "--whole-regions",
        action="store_true",
        help="--regions: write each region as one image of its own size, no grid inside it",
    )
    args = parser.parse_args()
    if args.whole_regions and not args.regions:
        sys.exit("--whole-regions needs --regions")
    if args.regions and not os.path.isfile(args.regions):
        sys.exit(f"--regions: no such file: {args.regions}")
    if args.tile < 1:
        sys.exit("--tile must be >= 1")
    if args.overlap < 0 or args.overlap >= args.tile:
        sys.exit(f"--overlap must be in [0, {args.tile})")
    if not 0 < args.black_frac <= 1:
        sys.exit("--black-frac must be in (0, 1]")
    if not 0 <= args.black_level <= 255:
        sys.exit("--black-level must be in [0, 255]")
    main(args)
