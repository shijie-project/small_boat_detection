"""LS -> COCO: ``tools/annotation/ls_to_coco.py``.

The step after annotating. Label Studio's *Export -> JSON* writes a list of
tasks with the boxes in percent; training reads a COCO file with the boxes in
pixels, keyed by bare file name. This converts the one into the other, and can
put the result straight into an existing dataset (``Merge into``) so the ids
continue where that file left off instead of clashing with it.

It reads a file and writes a file -- no GPU, no Label Studio server -- so it
runs in the ``data`` slot with the rest of the data prep.
"""

from ..core.jobs import DATA_SLOT
from ..core.paths import LS_COCO_SCRIPT, rel, resolve
from .base import (
    Feature,
    Field,
    JobSpec,
    existing_file,
    existing_files,
    flag,
    python_executable,
    text,
    whole_int,
)


DEFAULT_OUTPUT = "../data/annotations/new_coco.json"
SHIP_CATEGORY_ID = "3"  # AI-TOD's ship class -- what train_coco.json / val_coco.json use


def output_path(params, key="output"):
    """Where to write: inside the project, in a folder that exists."""
    value = text(params, key) or DEFAULT_OUTPUT
    path = resolve(value)
    if path is None:
        raise ValueError(f"output is outside the project: {value}")
    if not path.parent.is_dir():
        raise ValueError(f"output: folder does not exist: {value}")
    if path.is_dir():
        raise ValueError(f"output is a folder, not a file: {value}")
    return value


def images_dir(params, key="images"):
    """The optional images folder used to size tasks that have no boxes."""
    value = text(params, key)
    if not value:
        return ""
    path = resolve(value)
    if path is None or not path.is_dir():
        raise ValueError(f"images folder not found: {value}")
    return value


class LabelStudioCocoFeature(Feature):
    name = "ls-coco"
    label = "LS → COCO"
    slot = DATA_SLOT
    description = (
        "Turn one or more Label Studio **Export → JSON** files into the COCO file training "
        "reads. Boxes go from percent to pixels, `data.image` becomes the bare "
        "file name, cancelled annotations are dropped, and every box is labelled "
        "category 3 (*ship*) — the same shape as `../data/annotations/val_coco.json`. "
        "Images that ended up with no boxes are kept as background images. "
        "Pick several exports to fold them into one file — they are read in the "
        "order listed, so an image two of them share is taken from the later one. "
        "**Merge into** adds the result to an existing dataset instead: ids "
        "continue after that file's, and a task it already has is replaced. "
        "Splitting into train / val afterwards is "
        "`tools/annotation/random_split_coco.py`."
    )
    fields = [
        Field(
            "export",
            "Label Studio exports (-i)",
            kind="multichoice",
            source="ls_exports",
            prefer="/export/",
            info=(
                "Export → JSON (or JSON-MIN); pick as many as you like, oldest first. "
                "The `-info.json` beside one is metadata, not tasks."
            ),
        ),
        Field(
            "output",
            "Write to (-o)",
            value=DEFAULT_OUTPUT,
            info="Written via a temporary file, so merging into a file in place is safe.",
        ),
        [
            Field(
                "merge",
                "Merge into (--merge, optional)",
                kind="choice",
                source="coco",
                optional=True,
                empty_label="(start empty)",
                info="Add to this dataset; ids continue after its own.",
            ),
            Field(
                "on_duplicate",
                "Image it already has (--on-duplicate)",
                kind="choice",
                choices=("replace", "skip", "error"),
                value="replace",
            ),
        ],
        [
            Field("category_id", "Category id (--category-id)", value=SHIP_CATEGORY_ID),
            Field("category_name", "Category name (--category-name)", value="ship"),
        ],
        [
            Field("labels", "Keep only these labels (--labels)", info="Comma separated. Blank: every box."),
            Field("min_size", "Min box size in px (--min-size)", value="0", info="0 keeps every box."),
        ],
        [
            Field("skip_empty", "Drop images with no boxes (--skip-empty)", kind="flag"),
            Field("per_label", "One category per label (--per-label)", kind="flag"),
        ],
        [
            Field(
                "images",
                "Images folder (--images, optional)",
                info="Only for tasks with no boxes: their size is read from the file instead of assumed.",
            ),
            Field("dry_run", "Dry run (write nothing)", kind="flag"),
        ],
    ]

    def build(self, params):
        exports = existing_files(params, "export", "Label Studio export")
        output = output_path(params)
        cmd = [python_executable(params), LS_COCO_SCRIPT, "-o", output, "-i", *exports]

        merge = existing_file(params, "merge", "file to merge into", required=False)
        if merge:
            cmd += ["--merge", merge, "--on-duplicate", text(params, "on_duplicate") or "replace"]

        cmd += [
            "--category-id",
            str(whole_int(params, "category_id", int(SHIP_CATEGORY_ID), minimum=0)),
            "--category-name",
            text(params, "category_name") or "ship",
        ]
        labels = text(params, "labels")
        if labels:
            cmd += ["--labels", labels]
        min_size = whole_int(params, "min_size", 0, minimum=0)
        if min_size:
            cmd += ["--min-size", str(min_size)]
        images = images_dir(params)
        if images:
            cmd += ["--images", images]
        for name, option in (("skip_empty", "--skip-empty"), ("per_label", "--per-label"), ("dry_run", "--dry-run")):
            if flag(params, name):
                cmd.append(option)

        notes = []
        if merge and resolve(merge) == resolve(output):
            notes = [f"[webui] {rel(merge)} is being rewritten in place"]
        meta = {"feature": self.name, "outdir": rel(output), "cmd": " ".join(cmd)}
        return JobSpec(cmd, meta=meta, notes=notes)
