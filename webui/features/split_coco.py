"""Train / Val: ``tools/annotation/random_split_coco.py``.

The step after *LS -> COCO*, and the last one before training. Annotating
leaves one file covering everything (``all_coco.json``) beside one folder of
images (``images/all``); a config names ``images/train`` + ``train_coco.json``
and ``images/val`` + ``val_coco.json``, so they have to be cut in two first.

Like the other data tools it reads files and writes files -- no GPU -- so it
runs in the ``data`` slot rather than competing with training.
"""

from ..core.jobs import DATA_SLOT
from ..core.paths import SPLIT_SCRIPT, rel, resolve
from .base import Feature, Field, JobSpec, existing_file, flag, python_executable, text, whole_int


DEFAULT_VAL = "0.2"  # one image in five -- train : val = 4 : 1


def fraction(params, key, default):
    """A ratio field: a share of the dataset, so somewhere in [0, 1)."""
    raw = text(params, key) or default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number between 0 and 1, got {raw!r}") from None
    if not 0 <= value < 1:
        raise ValueError(f"{key} must be between 0 and 1, got {value}")
    return value


def out_dir(params, key, kind):
    """An output folder: blank means the script's own default (beside the input)."""
    value = text(params, key)
    if not value:
        return ""
    path = resolve(value)
    if path is None:
        raise ValueError(f"{kind} is outside the project: {value}")
    if path.exists() and not path.is_dir():
        raise ValueError(f"{kind} is a file, not a folder: {value}")
    if not path.parent.is_dir():
        raise ValueError(f"{kind}: parent folder does not exist: {value}")
    return value


def source_dir(params, key="images"):
    """The folder being split up -- it has to exist and hold the images."""
    value = text(params, key)
    if not value:
        raise ValueError("images folder is required")
    path = resolve(value)
    if path is None or not path.is_dir():
        raise ValueError(f"images folder not found: {value}")
    return value


class SplitFeature(Feature):
    # "Manual split" cuts scenes into tiles; this one cuts the dataset in two.
    name = "train-val"
    label = "Train / Val"
    slot = DATA_SLOT
    description = (
        "Cut one COCO file and its images into the **train / val** pair a config "
        "reads: `../data/images/train` + `train_coco.json` and `../data/images/val` "
        "+ `val_coco.json`. **Val share** `0.2` is one image in five — train : val "
        "of 4 : 1. The draw is stratified on whether an image has boxes, so the "
        "background images land in both splits in the same proportion, and **seed** "
        "makes it repeatable. **Mode** `link` hard-links instead of copying, which "
        "costs no extra disk; `move` empties the source folder, so it is the one to "
        "be sure about. Splitting again leaves the last split's files in place "
        "unless **Clean** is ticked."
    )
    fields = [
        [
            Field(
                "coco",
                "COCO file (--coco)",
                kind="choice",
                source="coco",
                prefer="all_coco",
                info="The one covering every image — what `LS → COCO` wrote.",
            ),
            Field(
                "images",
                "Images folder (--images)",
                kind="choice",
                source="dataset_dirs",
                prefer="/all$",
                info="The folder holding every image the file names.",
            ),
        ],
        [
            Field("val", "Val share (--val)", value=DEFAULT_VAL, info="0.2 = train : val of 4 : 1."),
            Field("test", "Test share (--test)", value="0.0", info="0 writes no test split at all."),
        ],
        [
            Field(
                "images_out",
                "Image folders written to (--images-out)",
                info="Blank: beside the images folder, so `../data/images/train` and `/val`.",
            ),
            Field(
                "annotations_out",
                "JSON written to (--annotations-out)",
                info="Blank: beside the COCO file, so `../data/annotations/train_coco.json`.",
            ),
        ],
        [
            Field("seed", "Seed (--seed)", value="42", info="Same seed, same split."),
            Field(
                "mode",
                "Images (--mode)",
                kind="choice",
                choices=("copy", "link", "move", "none"),
                value="copy",
                info="link: hard links, no extra disk. move: empties the source. none: json only.",
            ),
        ],
        [
            Field("clean", "Clean the split folders first (--clean)", kind="flag"),
            Field("dry_run", "Dry run (write nothing)", kind="flag"),
        ],
    ]

    def build(self, params):
        coco = existing_file(params, "coco", "COCO file (--coco)")
        mode = text(params, "mode") or "copy"
        images = source_dir(params) if mode != "none" else text(params, "images")

        val = fraction(params, "val", DEFAULT_VAL)
        test = fraction(params, "test", "0.0")
        if val + test >= 1:
            raise ValueError(f"val + test must leave something for train, got {val} + {test}")

        cmd = [python_executable(params), SPLIT_SCRIPT, "--coco", coco]
        if images:
            cmd += ["--images", images]
        cmd += ["--val", str(val), "--test", str(test), "--mode", mode]
        cmd += ["--seed", str(whole_int(params, "seed", 42, minimum=0))]

        images_out = out_dir(params, "images_out", "image output folder")
        if images_out:
            cmd += ["--images-out", images_out]
        annotations_out = out_dir(params, "annotations_out", "json output folder")
        if annotations_out:
            cmd += ["--annotations-out", annotations_out]
        for name, option in (("clean", "--clean"), ("dry_run", "--dry-run")):
            if flag(params, name):
                cmd.append(option)

        notes = []
        if mode == "move" and not flag(params, "dry_run"):
            notes = [f"[webui] --mode move: {rel(images)} is being emptied into the split folders"]
        outdir = rel(annotations_out) if annotations_out else rel(resolve(coco).parent)
        meta = {"feature": self.name, "outdir": outdir, "cmd": " ".join(cmd)}
        return JobSpec(cmd, meta=meta, notes=notes)
