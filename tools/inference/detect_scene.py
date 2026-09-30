"""One satellite scene in, its vessels out: tiling, detection and merging in one run.

The scene page (``webui/scene``) runs this for every detection; it also runs on
its own. It is the two scripts the annotation workflow runs one after the other
-- ``tools/dataset/tile_satellite.py`` to cut the scene, ``torch_inf_dir.py`` to
detect on the cut -- with nothing asked of a human in between:

  * the scene is cut into 1024 tiles at its original resolution, the right and
    bottom edge tiles padded with black; tiles with no image content (one flat
    colour, or 99% no-data black) are not written
  * every tile goes through the detector at full resolution, ``--batch`` at a
    time, and a box centred in the padding is dropped
  * the boxes are moved to scene coordinates, a boat cut by a tile border is
    joined back into one box, and a boat seen twice where tiles overlap is kept
    once (the same ``merge()`` as ``torch_inf_dir.py``)
  * with ``--mask``, a vessel whose centre lies inside one of the mask's
    polygons (buildings, a harbour, land) is set aside rather than reported.
    The pixels are not blacked out before detection: an artificial edge in the
    image would be a new place for false alarms, and a boat half in the mask
    would lose half its outline

Outputs, in ``-o``:

  detections.csv        one row per vessel: id, centre, size and box in scene px, score
  detections_coco.json  the same vessels as a COCO file of one image, the scene
  detections.json       per-tile and merged boxes, with the tile manifest's positions
  predictions.json      the per-tile boxes as COCO results (tile coordinates), what
                        the Label Studio import and the eval tools read
  overlay.jpg           the scene with the vessels drawn on it (longest side 4096 px),
                        and the mask's outline in yellow
  masked.json           the vessels the mask set aside, boxes in scene px (with --mask)
  tiles/<stem>/         the tiles and their ``tiles.json`` manifest
  run.json              what ran and what came out: model, settings, counts, timings

Lines starting ``[stage]`` mark the steps, for the page's progress bar.

Usage
-----
    python tools/inference/detect_scene.py -i ../data/satellite_images/<scene>.png \\
        -c configs/dfine/DFine-M-GoogleEarth.yml -r outputs/dfine_m_google_earth/<run>/best_stg1.pth \\
        -o ../data/scene_runs/<run id> [--mask mask.json]

A mask file is ``{"polygons": [[[x, y], ...], ...]}`` in scene pixels, each
polygon three points or more.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "dataset"))
import tile_satellite  # noqa: E402
import torch_inf_dir  # noqa: E402

TILE = 1024
CLASS_NAMES = {0: "ship"}  # the ship-only models label their one class 0


def stage(name):
    print(f"[stage] {name}", flush=True)


def load_mask(path):
    """The mask's polygons as lists of ``(x, y)``; none without a file."""
    if not path:
        return []
    with open(path, encoding="utf-8") as handle:
        polygons = json.load(handle).get("polygons", [])
    return [[(float(x), float(y)) for x, y in polygon] for polygon in polygons if len(polygon) >= 3]


def inside(x, y, polygon):
    """Even-odd rule: does a ray from ``(x, y)`` to the right cross the outline an odd number of times."""
    hit = False
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def masked_out(box, polygons):
    """Whether the centre of an ``xyxy`` box in scene px lies in the mask."""
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(inside(x, y, polygon) for polygon in polygons)


def apply_mask(records, merged, polygons):
    """``(kept, set aside)`` of the merged vessels; the per-tile records lose the same boxes."""
    if not polygons:
        return merged, []
    for record in records:
        if record["x"] is None:
            continue
        dx, dy = record["x"], record["y"]
        record["detections"] = [
            det
            for det in record["detections"]
            if not masked_out([v + (dx, dy)[i % 2] for i, v in enumerate(det["bbox_xyxy"])], polygons)
        ]
    kept = [det for det in merged if not masked_out(det["bbox_xyxy"], polygons)]
    aside = [det for det in merged if masked_out(det["bbox_xyxy"], polygons)]
    print(f"  mask: {len(polygons)} area(s), {len(aside)} of {len(merged)} vessels set aside")
    return kept, aside


def draw_mask(path, manifest, polygons):
    """The mask's outline on the overlay, at the overlay's scale."""
    if not polygons or not os.path.isfile(path):
        return
    image = Image.open(path)
    scale = image.width / manifest["width"]
    pen = ImageDraw.Draw(image)
    for polygon in polygons:
        pen.polygon([(x * scale, y * scale) for x, y in polygon], outline=(255, 200, 0), width=3)
    image.save(path, quality=90)


def cut(scene, out_root, overlap):
    """Tile the scene into ``out_root/<stem>/``; returns that folder and its manifest."""
    drop = tile_satellite.drop_test(skip_blank=True, drop_black=True, black_frac=0.99, black_level=8)
    tile_satellite.split_image(scene, out_root, TILE, overlap, ".png", 95, drop)
    folder = os.path.join(out_root, os.path.splitext(os.path.basename(scene))[0])
    manifest, _ = torch_inf_dir.load_manifest(folder)
    return folder, manifest


def vessels(merged):
    """The merged boxes as the rows the CSV and the page show, numbered by score."""
    rows = []
    for index, det in enumerate(merged, 1):
        x0, y0, x1, y1 = det["bbox_xyxy"]
        rows.append(
            {
                "id": index,
                "x": round((x0 + x1) / 2, 1),
                "y": round((y0 + y1) / 2, 1),
                "width": round(x1 - x0, 1),
                "height": round(y1 - y0, 1),
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "score": det["score"],
                "class": CLASS_NAMES.get(det["label"], str(det["label"])),
            }
        )
    return rows


def write_vessel_csv(path, rows):
    fields = ["id", "x", "y", "width", "height", "x0", "y0", "x1", "y1", "score", "class"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_scene_coco(path, scene, manifest, rows):
    """A COCO file of one image, the scene, with the vessels as scored annotations."""
    payload = {
        "images": [
            {
                "id": 1,
                "file_name": os.path.basename(scene),
                "width": manifest["width"],
                "height": manifest["height"],
            }
        ],
        "categories": [{"id": k, "name": v} for k, v in CLASS_NAMES.items()],
        "annotations": [
            {
                "id": row["id"],
                "image_id": 1,
                "category_id": 0,
                "bbox": [row["x0"], row["y0"], row["width"], row["height"]],
                "area": round(row["width"] * row["height"], 2),
                "score": row["score"],
                "iscrowd": 0,
            }
            for row in rows
        ],
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)


def main(args):
    started = time.time()
    os.makedirs(args.output, exist_ok=True)
    timings = {}
    settings = argparse.Namespace(  # what torch_inf_dir's functions read off its own command line
        config=args.config,
        resume=args.resume,
        batch=args.batch,
        device=args.device,
        thrh=args.thrh,
        classes=None,
        nms_iou=args.nms_iou,
        edge_tol=args.edge_tol,
        save_tiles="none",
        max_tiles=0,
    )

    stage("tiling")
    t = time.time()
    folder, manifest = cut(args.input, os.path.join(args.output, "tiles"), args.overlap)
    timings["tiling_s"] = round(time.time() - t, 1)
    if manifest is None or not manifest.get("tiles"):
        raise SystemExit("the scene gave no tiles with image content")

    stage("loading model")
    t = time.time()
    model, input_size = torch_inf_dir.build_model(args.config, args.resume, args.device)
    timings["model_s"] = round(time.time() - t, 1)

    stage("detecting")
    t = time.time()
    records = torch_inf_dir.run_scene(model, input_size, folder, args.output, settings, CLASS_NAMES)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    timings["detection_s"] = round(time.time() - t, 1)

    stage("merging")
    merged = torch_inf_dir.merge(records, settings)
    polygons = load_mask(args.mask)
    merged, aside = apply_mask(records, merged, polygons)
    rows = vessels(merged)

    stage("writing")
    t = time.time()
    torch_inf_dir.assign_image_ids(records, {}, 1)
    torch_inf_dir.write_predictions(os.path.join(args.output, "predictions.json"), records)
    torch_inf_dir.write_json(
        os.path.join(args.output, "detections.json"), folder, manifest, records, merged, settings, CLASS_NAMES
    )
    write_vessel_csv(os.path.join(args.output, "detections.csv"), rows)
    write_scene_coco(os.path.join(args.output, "detections_coco.json"), args.input, manifest, rows)
    torch_inf_dir.write_overlay(
        os.path.join(args.output, "overlay.jpg"), manifest, merged, "preview", 4096, CLASS_NAMES
    )
    draw_mask(os.path.join(args.output, "overlay.jpg"), manifest, polygons)
    if polygons:
        with open(os.path.join(args.output, "masked.json"), "w") as handle:
            json.dump([{"bbox_xyxy": det["bbox_xyxy"], "score": det["score"]} for det in aside], handle)
    timings["writing_s"] = round(time.time() - t, 1)
    timings["total_s"] = round(time.time() - started, 1)

    summary = {
        "status": "done",
        "finished": datetime.now().isoformat(timespec="seconds"),
        "scene": os.path.abspath(args.input).replace("\\", "/"),
        "scene_width": manifest["width"],
        "scene_height": manifest["height"],
        "config": args.config,
        "checkpoint": args.resume,
        "score_threshold": args.thrh,
        "overlap": args.overlap,
        "nms_iou": args.nms_iou,
        "tiles": len(records),
        "tiles_skipped": manifest.get("skipped_blank", 0) + manifest.get("skipped_black", 0),
        "vessels": len(rows),
        "mask_areas": len(polygons),
        "masked": len(aside),
        "timings": timings,
    }
    with open(os.path.join(args.output, "run.json"), "w") as handle:
        json.dump(summary, handle, indent=1)
    stage("done")
    print(f"{len(rows)} vessels ({len(aside)} masked) in {len(records)} tiles, {timings['total_s']}s -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="the scene image")
    parser.add_argument("-c", "--config", required=True, help="model config yml")
    parser.add_argument("-r", "--resume", required=True, help="checkpoint .pth")
    parser.add_argument("-o", "--output", required=True, help="run folder to write into")
    parser.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--thrh", type=float, default=0.5, help="score threshold (default 0.5)")
    parser.add_argument("--overlap", type=int, default=0, help="px shared by neighbouring tiles (default 0)")
    parser.add_argument("--batch", type=int, default=8, help="tiles per forward pass")
    parser.add_argument("--nms_iou", type=float, default=0.5, help="IoU for removing duplicates")
    parser.add_argument("--mask", default=None, help="json of polygons to leave out, in scene px")
    parser.add_argument("--edge-tol", type=int, default=2, help="px of slack when joining a boat cut by a border")
    main(parser.parse_args())
