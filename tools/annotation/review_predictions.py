"""Run a detector over a folder of images and put its disagreements with the GT
into Label Studio as ``ship-pred`` boxes, to correct the GT against.

For every image in the folder:

  * the image is in the GT json: each prediction is compared with the image's
    GT boxes, and one whose best IoU with any of them is below ``--iou``
    (default 0.75) -- a box the GT has drawn differently, or not at all -- is
    added. A prediction that agrees with a GT box is not.
  * the image is not in the GT json (or no json is given): every prediction
    is added.

The boxes go into the task's existing annotation, next to the GT ``ship`` boxes,
so the two can be compared and the GT adjusted in place; a task with no
annotation gets a new one. Running again replaces the ``ship-pred`` boxes an
earlier run left on the same images -- nothing else in the annotation is
touched, and images outside the folder are not looked at. ``--undo`` strips
every ``ship-pred`` box from the project.

Settings (``LABEL_STUDIO_URL`` / ``_API_KEY`` / ``_PROJECT_ID`` and the control
names) come from the first ``.env`` found, as for ``predictions_to_annotations.py``.

Usage
-----
    python tools/annotation/review_predictions.py \\
        -c configs/dome/Dome-M-AEA.yml -r ../ckpts/Dome-M-AEA-best.pth \\
        -i ../data/annotated/all --gt ../data/annotated/all.json --dry-run

    # remove the ship-pred boxes again
    python tools/annotation/review_predictions.py --undo
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.ops import box_iou


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "tools" / "inference"))

from predictions_to_annotations import (  # noqa: E402  (same folder)
    ENV_CANDIDATES,
    connect,
    fetch_tasks,
    field,
    load_env,
    load_json,
    project_labels,
    setting,
    task_basename,
    to_region,
)
from torch_inf_dir import IMAGE_SUFFIXES, build_model, filter_dets, prepare  # noqa: E402


DEFAULT_LABEL = "ship-pred"


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def load_gt(path):
    """``{file name: [x1, y1, x2, y2] boxes}`` from a COCO file, keyed by base name."""
    data = load_json(path)
    names = {int(image["id"]): Path(image["file_name"]).name for image in data.get("images", [])}
    boxes = {name: [] for name in names.values()}
    for ann in data.get("annotations", []):
        name = names.get(int(ann.get("image_id", -1)))
        if name is None or "bbox" not in ann:
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        boxes[name].append([x, y, x + w, y + h])
    return boxes


def disagreeing(pred_boxes, gt_boxes, iou_threshold):
    """Mask of the predictions whose best IoU with any GT box is below the threshold."""
    if not len(pred_boxes):
        return torch.zeros(0, dtype=torch.bool)
    if not gt_boxes:
        return torch.ones(len(pred_boxes), dtype=torch.bool)
    best = box_iou(pred_boxes, torch.tensor(gt_boxes, dtype=pred_boxes.dtype)).max(dim=1).values
    return best < iou_threshold


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


@torch.no_grad()
def predict(model, input_size, paths, args):
    """``{file name: (width, height, xyxy boxes [N, 4], scores [N])}`` for every image."""
    to_tensor = T.Compose([T.Resize((input_size, input_size)), T.ToTensor()])
    results = {}
    batch, meta = [], []
    started = time.time()

    def flush():
        if not batch:
            return
        sizes = torch.tensor([[s, s] for _, s, _, _ in meta], device=args.device, dtype=torch.float32)
        labels, boxes, scores = model(torch.cat(batch), sizes)
        for i, (name, _, width, height) in enumerate(meta):
            b, s, _ = filter_dets(boxes[i], scores[i], labels[i], args.thrh, args.classes, width, height)
            results[name] = (width, height, b.float(), s.float())
        batch.clear()
        meta.clear()

    for index, path in enumerate(paths, 1):
        im = Image.open(path).convert("RGB")
        canvas, canvas_size, width, height = prepare(im, input_size)
        batch.append(to_tensor(canvas).unsqueeze(0).to(args.device))
        meta.append((path.name, canvas_size, width, height))
        if len(batch) >= args.batch:
            flush()
        if index % 50 == 0 or index == len(paths):
            print(f"  {index}/{len(paths)} images")
    flush()
    total = sum(len(r[2]) for r in results.values())
    print(f"  {total} detection(s) above score {args.thrh} in {time.time() - started:.1f}s")
    return results


# --------------------------------------------------------------------------- #
# Label Studio
# --------------------------------------------------------------------------- #
def is_review_region(region, label):
    value = field(region, "value", {}) or {}
    return field(region, "type") == "rectanglelabels" and label in (field(value, "rectanglelabels", []) or [])


def without_review(results, label):
    """The annotation's results minus our boxes -- and minus the per-region
    attributes (choices sharing the box's id) that were attached to them."""
    stale = {field(r, "id") for r in results if is_review_region(r, label)}
    return [r for r in results if field(r, "id") not in stale], bool(stale)


def sync(client, task, regions, label, dry_run):
    """Make the task's first annotation hold exactly ``regions`` as its ``label`` boxes.

    Returns "created", "updated", "cleared" or "unchanged".
    """
    annotations = field(task, "annotations", []) or []
    if not annotations:
        if not regions:
            return "unchanged"
        if not dry_run:
            client.annotations.create(id=int(field(task, "id")), result=regions)
        return "created"

    target = annotations[0]
    kept, had_stale = without_review(list(field(target, "result", []) or []), label)
    if not regions and not had_stale:
        return "unchanged"
    if not dry_run:
        merged = kept + regions
        if merged:
            client.annotations.update(id=int(field(target, "id")), result=merged)
        else:  # the annotation only ever held our boxes
            client.annotations.delete(id=int(field(target, "id")))
    return "updated" if regions else "cleared"


def undo(client, tasks, label, dry_run):
    updated = deleted = 0
    for task in tasks:
        for annotation in field(task, "annotations", []) or []:
            kept, had_stale = without_review(list(field(annotation, "result", []) or []), label)
            if not had_stale:
                continue
            if kept:
                updated += 1
                if not dry_run:
                    client.annotations.update(id=int(field(annotation, "id")), result=kept)
            else:
                deleted += 1
                if not dry_run:
                    client.annotations.delete(id=int(field(annotation, "id")))
    verb = "would strip" if dry_run else "stripped"
    print(f"undo: {verb} `{label}` boxes from {updated} annotation(s) and {deleted} annotation(s) that held only them")


# --------------------------------------------------------------------------- #
def main(args):
    env_file = load_env(args.env)
    print(f"settings from {env_file or 'the environment'}")
    project_id = args.project or int(setting("LABEL_STUDIO_PROJECT_ID", "0"))
    if project_id <= 0:
        raise SystemExit("no project: pass --project or set LABEL_STUDIO_PROJECT_ID")

    # Label Studio first: no point in running the model if it is not reachable
    client = connect()
    project, tasks = fetch_tasks(client, project_id)
    from_name = setting("LABEL_STUDIO_FROM_NAME", "label")
    labels = project_labels(project, from_name)
    if labels and args.label not in labels:
        raise SystemExit(f"label {args.label!r} is not in the project's config; it offers: {', '.join(labels)}")

    if args.undo:
        undo(client, tasks, args.label, args.dry_run)
        if args.dry_run:
            print("dry run: nothing was written")
        return

    paths = list_images(args.input)
    if not paths:
        raise SystemExit(f"no images in {args.input}")
    gt = load_gt(args.gt) if args.gt else {}
    if args.gt:
        print(f"GT: {sum(len(v) for v in gt.values())} box(es) over {len(gt)} image(s) in {args.gt}")
    else:
        print("no GT json: every prediction is added")

    print(f"loading {args.resume}")
    model, input_size = build_model(args.config, args.resume, args.device)
    print(f"{len(paths)} image(s) in {args.input}, input size {input_size}")
    predictions = predict(model, input_size, paths, args)

    to_name = setting("LABEL_STUDIO_TO_NAME", "image")
    image_key = setting("LABEL_STUDIO_DATA_IMAGE_KEY", "image")
    tasks_by_name = {}
    for task in tasks:
        name = task_basename(task, image_key)
        if name:
            tasks_by_name.setdefault(name, task)

    counts = defaultdict(int)
    added = matched = 0
    no_task = []
    for name, (width, height, boxes, scores) in sorted(predictions.items()):
        in_gt = name in gt
        counts["images in GT" if in_gt else "images not in GT"] += 1
        keep = disagreeing(boxes, gt.get(name, []), args.iou)
        matched += int((~keep).sum())
        regions = []
        for box in boxes[keep].tolist():
            x1, y1, x2, y2 = box
            region = to_region([x1, y1, x2 - x1, y2 - y1], width, height, args.label, from_name, to_name)
            if region is not None:
                regions.append(region)

        task = tasks_by_name.get(name)
        if task is None:
            if regions:
                no_task.append(name)
            continue
        status = sync(client, task, regions, args.label, args.dry_run)
        counts[f"tasks {status}"] += 1
        added += len(regions)
        if regions and args.verbose:
            print(f"  {name}: {len(regions)} `{args.label}` box(es){'' if in_gt else ' (not in GT)'}")

    verb = "would add" if args.dry_run else "added"
    print(f"{verb} {added} `{args.label}` box(es); {matched} prediction(s) agreed with GT (IoU >= {args.iou})")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    if no_task:
        print(f"  {len(no_task)} image(s) with boxes to add have no task in the project, e.g. {no_task[0]}")
    if args.dry_run:
        print("dry run: nothing was written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", help="model config yml")
    parser.add_argument("-r", "--resume", help="checkpoint .pth")
    parser.add_argument("-i", "--input", default="../data/annotated/all", help="folder of images")
    parser.add_argument("--gt", default=None, help="COCO json with the GT boxes to compare against (optional)")
    parser.add_argument(
        "--iou", type=float, default=0.75, help="add a prediction whose best IoU with GT is below this"
    )
    parser.add_argument("--thrh", type=float, default=0.4, help="score threshold")
    classes = parser.add_mutually_exclusive_group()
    classes.add_argument("--classes", type=int, nargs="+", default=[3], help="keep only these labels (default: 3)")
    classes.add_argument("--all-classes", action="store_true", help="keep every class")
    parser.add_argument("--batch", type=int, default=8, help="images per forward pass")
    parser.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--label", default=DEFAULT_LABEL, help=f"rectanglelabel for the added boxes ({DEFAULT_LABEL})")
    parser.add_argument("--dry-run", action="store_true", help="report what would happen; write nothing")
    parser.add_argument("--undo", action="store_true", help="strip every --label box from the project instead")
    parser.add_argument("--verbose", action="store_true", help="one line per image that gets boxes")
    parser.add_argument("--project", type=int, default=0, help="project id (default: LABEL_STUDIO_PROJECT_ID)")
    parser.add_argument(
        "--env", default=None, help=f"the .env to read (default: first of {', '.join(ENV_CANDIDATES)})"
    )
    args = parser.parse_args()
    if args.all_classes:
        args.classes = None
    if not args.undo:
        if not (args.config and args.resume):
            sys.exit("-c/--config and -r/--resume are required (or use --undo)")
        if not os.path.isdir(args.input):
            sys.exit(f"no such folder: {args.input}")
    main(args)
