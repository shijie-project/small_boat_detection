"""LS review: ``tools/annotation/review_predictions.py``.

Pick a model, run it over a folder of annotated images, and send the boxes it
disagrees with the GT on to Label Studio as ``ship-pred`` -- in the same
annotation as the GT ``ship`` boxes, so the GT can be corrected against them.
A prediction whose best IoU with the image's GT boxes is below the threshold
goes over; one on an image the GT json does not list (or with no json at all)
always does.

It runs the model, so it sits in the ``run`` slot with the other GPU work.
Like *LS import* it writes into the live project, so **dry run is on by
default**: the first click reports what would change and writes nothing.
"""

from ..core.paths import LS_REVIEW_SCRIPT, rel, resolve
from .base import (
    Feature,
    Field,
    JobSpec,
    config_path,
    existing_file,
    flag,
    gpu_env,
    positive_int,
    python_executable,
    text,
)
from .inference import class_args, fraction


DEFAULT_IMAGES = "../data/annotated/all"
DEFAULT_GT = "../data/annotated/all.json"


def image_folder(params):
    value = text(params, "images")
    if not value:
        raise ValueError("image folder is required")
    path = resolve(value)
    if path is None:
        raise ValueError(f"image folder is outside the project: {value}")
    if not path.is_dir():
        raise ValueError(f"image folder not found: {value}")
    return value, path


class LabelStudioReviewFeature(Feature):
    name = "ls-review"
    label = "LS review"
    description = (
        "Run a model over a folder of images and add its boxes to Label Studio as "
        "`ship-pred`, next to the GT `ship` boxes, to correct the GT against. For an "
        "image in the GT json, only predictions whose best IoU with its GT boxes is "
        "below the threshold are added; for an image not in it (or with no json), "
        "every prediction is. Re-running replaces the `ship-pred` boxes an earlier "
        "run left on those images. **Dry run is ticked** — untick it to write. Label "
        "Studio must be running (*Label Studio* tab); settings come from `../data/.env`."
    )
    fields = [
        [
            Field("images", "Image folder (-i)", value=DEFAULT_IMAGES),
            Field(
                "gt",
                "GT json (--gt, optional)",
                value=DEFAULT_GT,
                info="COCO file to compare against. Blank: no comparison, every prediction is added.",
            ),
        ],
        Field(
            "config",
            "Config (-c)",
            kind="choice",
            source="configs",
            value="configs/dome/Dome-M-AEA.yml",
            prefer="AEA",
        ),
        Field(
            "checkpoint",
            "Checkpoint (-r, required)",
            kind="choice",
            source="checkpoints",
            prefer="AEA-best",
        ),
        [
            Field(
                "iou",
                "IoU threshold (--iou)",
                value="0.75",
                info="A prediction below this IoU with every GT box is added.",
            ),
            Field("thrh", "Score threshold (--thrh)", value="0.4"),
        ],
        [
            Field("classes", "Classes (--classes)", value="3", info="Blank keeps every class."),
            Field("batch", "Images per forward pass (--batch)", value="8"),
        ],
        [
            Field("gpus", "GPUs (CUDA_VISIBLE_DEVICES)", value="0", info="e.g. 0"),
            Field("device", "Device (-d)", kind="choice", choices=("cuda", "cpu"), value="cuda"),
        ],
        [
            Field("dry_run", "Dry run (write nothing)", kind="flag", value="1"),
            Field(
                "undo",
                "Remove ship-pred boxes instead",
                kind="flag",
                info="Strip every `ship-pred` box from the project; ignores the model and folder.",
            ),
        ],
        Field("project", "Project id (optional)", info="Blank: LABEL_STUDIO_PROJECT_ID from the .env."),
    ]

    def build(self, params):
        cmd = [python_executable(params), LS_REVIEW_SCRIPT]
        outdir = ""

        if flag(params, "undo"):
            cmd.append("--undo")
        else:
            images, path = image_folder(params)
            config = config_path(params)
            checkpoint = existing_file(params, "checkpoint", "checkpoint (-r)")
            try:
                gt = existing_file(params, "gt", "GT json", required=False)
            except ValueError as exc:
                raise ValueError(f"{exc} (clear the field to skip the comparison)") from None
            cmd += [
                "-c",
                config,
                "-r",
                checkpoint,
                "-i",
                images,
                "--iou",
                fraction(params, "iou", 0.75),
                "--thrh",
                fraction(params, "thrh", 0.4),
                "--batch",
                str(positive_int(params, "batch", 8)),
                "-d",
                text(params, "device") or "cuda",
            ]
            cmd += class_args(params)
            if gt:
                cmd += ["--gt", gt]
            outdir = rel(path)

        if flag(params, "dry_run"):
            cmd.append("--dry-run")
        project = text(params, "project")
        if project:
            if not project.isdigit():
                raise ValueError(f"project id must be a number, got {project!r}")
            cmd += ["--project", project]

        meta = {"feature": self.name, "outdir": outdir, "cmd": " ".join(cmd)}
        return JobSpec(cmd, env=gpu_env(params), meta=meta)
