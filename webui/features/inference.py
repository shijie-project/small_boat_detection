"""Inference: ``tools/inference/torch_inf_dir.py`` over a folder of cut tiles.

The step after **Manual split**. Pick the ``split_images/<scene>/`` folder the
picker wrote, pick a checkpoint, and the detector runs over every tile in it and
stitches the boxes back onto the full scene (``tiles.json`` says where each tile
came from). Results land in ``<folder>/inf_det/``: a ``predictions.json`` in the
COCO results format the eval and Label Studio tools already read, plus the
merged full-image boxes and the drawings.

Pointing at ``split_images/`` itself rather than one scene runs the lot -- the
script's ``--all``, which the form supplies for you, since a job started from a
browser has no stdin to answer the script's "which one?" prompt with.

It wants the GPU, so it sits in the ``run`` slot with train and test: one at a
time, rather than an inference sweep quietly stealing memory from a training
run.
"""

from ..core.discovery import IMAGE_SUFFIXES
from ..core.paths import INFER_SCRIPT, rel, resolve
from .base import (
    Feature,
    Field,
    JobSpec,
    config_path,
    existing_file,
    gpu_env,
    positive_int,
    python_executable,
    text,
    whole_int,
)


OUT_DIRNAME = "inf_det"  # must match torch_inf_dir.py
DEFAULT_COCO = "../data/annotations/val_coco.json"


def has_tiles(path):
    """True if this folder holds tile images itself (rather than scene folders)."""
    return any(
        entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES and "_det" not in entry.name
        for entry in path.iterdir()
    )


def tile_folder(params):
    """``(value, path, every)`` -- the folder to run on and whether that means all of them."""
    value = text(params, "tiles")
    if not value:
        raise ValueError("tile folder is required -- cut a scene with Manual split first")
    path = resolve(value)
    if path is None:
        raise ValueError(f"tile folder is outside the project: {value}")
    if not path.is_dir():
        raise ValueError(f"not a folder: {value}")
    if has_tiles(path):
        return value, path, False
    if any(child.is_dir() and has_tiles(child) for child in path.iterdir()):
        return value, path, True
    raise ValueError(f"no tiles in {value} -- cut the scene with Manual split first")


def class_args(params):
    """``--classes 3`` from the form, or ``--all-classes`` when it is left blank."""
    raw = text(params, "classes").replace(",", " ").split()
    if not raw:
        return ["--all-classes"]
    for value in raw:
        if not value.lstrip("-").isdigit():
            raise ValueError(f"classes must be numbers, got {value!r}")
    return ["--classes", *raw]


def fraction(params, key, default):
    """A 0-1 form value (score threshold, NMS IoU)."""
    raw = text(params, key) or str(default)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None
    if not 0 <= value <= 1:
        raise ValueError(f"{key} must be between 0 and 1, got {value}")
    return f"{value:g}"


class InferenceFeature(Feature):
    name = "inference"
    label = "Inference"
    description = (
        "Detect boats in a folder of tiles. Every tile goes through the net at its "
        "original resolution, the boxes are shifted back into full-image coordinates "
        "using `tiles.json`, boats cut by a tile border are stitched, and duplicates "
        "go to a global NMS. Output in `<folder>/inf_det/`: `predictions.json` (COCO "
        "results, per tile — what `per_image_metrics.py` and the Label Studio import "
        "read), `detections.json` (per tile **and** merged into full-image "
        "coordinates), `detections.csv`, the tiles that hit with boxes drawn, and an "
        "overlay of the whole scene."
    )
    fields = [
        Field(
            "tiles",
            "Tile folder (-i)",
            kind="choice",
            source="tiles",
            prefer="split_images/",
            info="A `split_images/<scene>/` folder; picking `split_images/` itself runs every scene.",
        ),
        Field(
            "config",
            "Config (-c)",
            kind="choice",
            source="configs",
            value="configs/dome/Dome-M-AEA.yml",
            prefer="AEA",
            info="Its eval_spatial_size sets the input size (AEA 1024, AITOD 800).",
        ),
        Field(
            "checkpoint",
            "Checkpoint (-r, required)",
            kind="choice",
            source="checkpoints",
            prefer="AEA-best",
        ),
        [
            Field("thrh", "Score threshold (--thrh)", value="0.4"),
            Field("batch", "Tiles per forward pass (--batch)", value="8"),
        ],
        [
            Field("classes", "Classes (--classes)", value="3", info="Blank keeps every class."),
            Field(
                "coco",
                "Annotations json (--coco)",
                value=DEFAULT_COCO,
                info="Class names, and the image_id a listed tile keeps in predictions.json.",
            ),
        ],
        [
            Field(
                "save_tiles",
                "Save tiles with boxes",
                kind="choice",
                choices=("hits", "all", "none"),
                value="hits",
            ),
            Field(
                "overlay",
                "Scene overlay",
                kind="choice",
                choices=("preview", "full", "none"),
                value="preview",
                info="preview: downscaled to 4096 px.",
            ),
        ],
        [
            Field("nms_iou", "Global NMS IoU (--nms_iou)", value="0.5"),
            Field(
                "edge_tol",
                "Border stitch slack px (--edge-tol)",
                value="2",
                info="-1 turns the stitching off.",
            ),
        ],
        [
            Field("gpus", "GPUs (CUDA_VISIBLE_DEVICES)", value="0", info="e.g. 0"),
            Field("device", "Device (-d)", kind="choice", choices=("cuda", "cpu"), value="cuda"),
        ],
        [
            Field("output", "Output root (-o, optional)", info=f"Blank: <tile folder>/{OUT_DIRNAME}/"),
            Field("max_tiles", "Limit tiles (--max-tiles)", info="Blank: all of them. For a quick look."),
        ],
    ]

    def build(self, params):
        tiles, path, every = tile_folder(params)
        config = config_path(params)
        checkpoint = existing_file(params, "checkpoint", "checkpoint (-r)")
        coco = existing_file(params, "coco", "annotations json", required=False)

        cmd = [
            python_executable(params),
            INFER_SCRIPT,
            "-c",
            config,
            "-r",
            checkpoint,
            "-i",
            tiles,
            "-d",
            text(params, "device") or "cuda",
            "--batch",
            str(positive_int(params, "batch", 8)),
            "--thrh",
            fraction(params, "thrh", 0.4),
            "--nms_iou",
            fraction(params, "nms_iou", 0.5),
            "--edge-tol",
            str(whole_int(params, "edge_tol", 2, minimum=-1)),
            "--save-tiles",
            text(params, "save_tiles") or "hits",
            "--overlay",
            text(params, "overlay") or "preview",
        ]
        cmd += class_args(params)
        if every:  # no stdin behind a browser, so never let the script ask
            cmd.append("--all")
        if coco:
            cmd += ["--coco", coco]

        output = text(params, "output")
        if output:
            if resolve(output) is None:
                raise ValueError(f"output root is outside the project: {output}")
            cmd += ["-o", output]
        if text(params, "max_tiles"):
            cmd += ["--max-tiles", str(positive_int(params, "max_tiles", 1))]

        outdir = rel(output) if output else f"{rel(path)}/{'*/' if every else ''}{OUT_DIRNAME}"
        meta = {"feature": self.name, "config": config, "outdir": outdir, "cmd": " ".join(cmd)}
        return JobSpec(cmd, env=gpu_env(params), meta=meta)
