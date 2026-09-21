"""Filesystem layout shared by every feature.

The package lives in ``<project>/webui``, so the project root -- the directory
every job has to run from and every path in the UI is relative to -- is the
parent of the package.
"""

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_DIR.parent
STATIC_DIR = PACKAGE_DIR / "static"

CONFIG_DIR = ROOT / "configs" / "dome"
CKPT_DIRS = ROOT.parent / "ckpts"
DOME_CKPT_DIR = ROOT / "dome_ckpts"  # the Dome pretrained weights training fine-tunes from
DATA_DIRS = ROOT.parent / "data"
SATELLITE_DIR = DATA_DIRS / "satellite_images"
IMAGES_DIR = DATA_DIRS / "images"
TRAIN_SCRIPT = "train.py"
INFER_SCRIPT = "tools/inference/torch_inf_dir.py"
LS_IMPORT_SCRIPT = "tools/annotation/predictions_to_annotations.py"
LS_REVIEW_SCRIPT = "tools/annotation/review_predictions.py"
LS_COCO_SCRIPT = "tools/annotation/ls_to_coco.py"
SPLIT_SCRIPT = "tools/annotation/random_split_coco.py"
PICKER_SCRIPT = "tools/dataset/split_picker.py"

# The trees the UI lists from, and therefore the only ones it may hand back.
ALLOWED_ROOTS = (ROOT, CKPT_DIRS, DATA_DIRS)

HOST = os.environ.get("WEBUI_HOST", "127.0.0.1")
PORT = int(os.environ.get("WEBUI_PORT", "8000"))


def rel(path) -> str:
    """Path as the UI shows it -- relative to the project root where possible.

    Checkpoints and datasets sit next to the project, so they come out as
    ``../ckpts/...`` / ``../data/...``, the same way the configs reference them;
    anything further out stays absolute.
    """
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    for base, prefix in ((ROOT, ""), (ROOT.parent, "../")):
        try:
            return prefix + str(resolved.relative_to(base)).replace("\\", "/")
        except ValueError:
            continue
    return str(resolved).replace("\\", "/")


def resolve(path):
    """Turn a browser-supplied path back into a real one, or ``None``.

    Relative paths are read from the project root -- the directory jobs run in.
    Anything outside :data:`ALLOWED_ROOTS` is refused: the page only sends back
    paths we listed ourselves, so anything else is a bug or a probe.
    """
    if not path:
        return None
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    if any(resolved == base or base in resolved.parents for base in ALLOWED_ROOTS):
        return resolved
    return None
