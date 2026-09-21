"""Discovery of the configs / checkpoints that fill the dropdowns.

Every feature picks from the same two lists, so they are collected once here
and served through a single ``/api/options`` endpoint.
"""

import os
import sys

from .paths import CKPT_DIRS, CONFIG_DIR, DATA_DIRS, IMAGES_DIR, ROOT, SATELLITE_DIR, rel


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
SPLIT_DIRNAME = "split_images"  # must match tile_satellite.py
OUT_DIRNAME = "inf_det"  # must match torch_inf_dir.py: holds the predictions we import
# Output of the picker and of tiled inference -- thousands of files, never inputs.
SKIP_DIRS = (SPLIT_DIRNAME, "crops")


def generated(name):
    """A folder we wrote ourselves: our own output is never an input.

    Hidden folders go too -- the picker's preview cache lives in one.
    """
    return name in SKIP_DIRS or "_det" in name or name.startswith(".")


def list_configs():
    if not CONFIG_DIR.is_dir():
        return []
    return [rel(p) for p in sorted(CONFIG_DIR.glob("*.yml"))]


def list_checkpoints():
    """The weights worth picking: a run's ``best_stg*.pth`` and the kept ``*-best.pth``.

    The second half is what inference actually runs -- the checkpoints promoted
    out of a run folder and given a name (``Dome-M-AEA-best.pth``).
    """
    out = []
    if CKPT_DIRS.is_dir():
        out.extend(
            rel(p) for p in sorted(CKPT_DIRS.rglob("*.pth")) if p.stem.startswith("best_") or p.stem.endswith("-best")
        )
    return out


def list_satellite_dirs(limit=200):
    """What the picker can be pointed at: the whole folder first, then each one holding images.

    The picker takes a folder, not an image, and lists every scene under it. The
    purchased imagery arrives nested (``zip files/<order>/<image>.tif``), so this
    walks the tree -- minus the folders we generate ourselves, which hold
    hundreds of crops and are never an input.
    """
    if not SATELLITE_DIR.is_dir():
        return []
    found = [rel(SATELLITE_DIR)]
    for root, dirs, files in os.walk(SATELLITE_DIR):
        dirs[:] = sorted(d for d in dirs if not generated(d))
        if root != str(SATELLITE_DIR) and any(f.lower().endswith(IMAGE_SUFFIXES) for f in files):
            found.append(rel(root))
        if len(found) >= limit:
            break
    return found


def holds_images(path):
    """True if this folder has image files of its own."""
    try:
        return any(name.lower().endswith(IMAGE_SUFFIXES) for name in os.listdir(path))
    except OSError:
        return False


def list_tile_dirs(limit=200):
    """What inference can be pointed at: each ``split_images/<scene>/``, root first.

    The root stands for "every scene under it" -- the inference script's
    ``--all``. Our own ``inf_det/`` output is not an input, so it never shows up.
    """
    if not SATELLITE_DIR.is_dir():
        return []
    found = []
    for root, dirs, _ in os.walk(SATELLITE_DIR):
        if os.path.basename(root) == SPLIT_DIRNAME:
            scenes = [d for d in sorted(dirs) if not generated(d) and holds_images(os.path.join(root, d))]
            if scenes:
                found.append(rel(root))
                found += [rel(os.path.join(root, d)) for d in scenes]
            dirs[:] = []  # the scenes themselves hold nothing but tiles
        else:  # descend, but only into split_images once we reach it
            dirs[:] = sorted(d for d in dirs if not generated(d) or d == SPLIT_DIRNAME)
        if len(found) >= limit:
            break
    return found


def list_prediction_files(limit=200):
    """COCO results files worth importing: every inference run's, then the split ones.

    The inference runs come first because they are the ones written from the
    dashboard; ``<split>_preds.json`` is what ``train.py --test-only`` leaves in
    the annotations folder.
    """
    found = []
    if SATELLITE_DIR.is_dir():
        for root, dirs, files in os.walk(SATELLITE_DIR):
            dirs[:] = sorted(d for d in dirs if not generated(d) or d in (SPLIT_DIRNAME, OUT_DIRNAME))
            if "predictions.json" in files:
                found.append(rel(os.path.join(root, "predictions.json")))
            if len(found) >= limit:
                return found
    annotations = DATA_DIRS / "annotations"
    if annotations.is_dir():
        found += [rel(p) for p in sorted(annotations.glob("*_preds.json"))]
    return found[:limit]


def annotation_files(pattern="*.json"):
    """Files in ``../data/annotations``, newest first -- it is a working folder."""
    folder = DATA_DIRS / "annotations"
    if not folder.is_dir():
        return []
    return sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def list_ls_exports(limit=40):
    """Label Studio exports worth converting: the export folder, then annotations.

    ``../data/export`` is where Label Studio's own *Export* button writes
    (``project-10-at-....json``, with a ``-info.json`` beside it that is metadata,
    not tasks); anything hand-copied tends to land in ``../data/annotations``.
    Our own COCO files live there too, so they are filtered back out.
    """
    found = []
    exports = DATA_DIRS / "export"
    if exports.is_dir():
        files = (p for p in exports.glob("*.json") if not p.name.endswith("-info.json"))
        found += [rel(p) for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)]
    found += [rel(p) for p in annotation_files() if not p.stem.endswith(("_coco", "_preds", "_fixed", "-info"))]
    return found[:limit]


def list_dataset_dirs(limit=40):
    """The training-set image folders: ``../data/images/<split>``, the full one first.

    ``all`` is what a split is drawn from and the others are what it writes, so
    the folder holding everything leads the list.
    """
    if not IMAGES_DIR.is_dir():
        return []
    folders = [p for p in sorted(IMAGES_DIR.iterdir()) if p.is_dir() and holds_images(p)]
    folders.sort(key=lambda p: (p.name not in ("all", "full"), p.name))
    return [rel(p) for p in folders][:limit]


def list_coco_files(limit=40):
    """The COCO annotation files a conversion can be merged into."""
    return [rel(p) for p in annotation_files("*_coco.json")][:limit]


def options():
    """Everything the page needs to build its forms."""
    return {
        "configs": list_configs(),
        "checkpoints": list_checkpoints(),
        "satellite": list_satellite_dirs(),
        "tiles": list_tile_dirs(),
        "predictions": list_prediction_files(),
        "ls_exports": list_ls_exports(),
        "coco": list_coco_files(),
        "dataset_dirs": list_dataset_dirs(),
        "python": sys.executable,
        "root": str(ROOT),
    }
