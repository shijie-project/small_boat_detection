"""Manual split: ``tools/dataset/split_picker.py``, embedded in its tab.

Everything else here is a form, because everything else *is* a form. Choosing
which 1024 cells of a 122 MP scene to cut is not: it needs the image at real
resolution, zoomed and panned, with the grid drawn on it. So this feature starts
a small web app that serves the scene like a map, and the tab shows that app in
an iframe -- the picker owns its own canvas and never goes through Gradio's
event loop, which is what made the previous attempt unusable.

It is a server, so it stays up until Stop, and it gets its own slot: leaving the
picker open must not block Label Studio or the data prep.
"""

import gradio as gr

from ..core.jobs import PICK_SLOT
from ..core.paths import PICKER_SCRIPT, SATELLITE_DIR, rel, resolve
from .base import Feature, Field, JobSpec, positive_int, python_executable, text


DEFAULT_PORT = 8011
FRAME_ID = "manual-split-frame"


def folder(params, key, kind, required=True):
    """One of the two folder fields, kept inside the trees the UI may touch."""
    value = text(params, key)
    if not value:
        if required:
            raise ValueError(f"{kind} is required")
        return ""
    path = resolve(value)
    if path is None:
        raise ValueError(f"{kind} is outside the project: {value}")
    if required and not path.is_dir():
        raise ValueError(f"{kind} not found: {value}")
    return value


class ManualSplitFeature(Feature):
    name = "manual-split"
    label = "Manual split"
    slot = PICK_SLOT
    wide = True  # the canvas wants the page, not a third of it
    description = (
        "Pick the cells worth cutting on the scene itself: scroll to zoom, drag with the "
        "right button to pan, click or sweep with the left to take cells, **Apply** cuts "
        "exactly those. Zoom past the preview and each cell is re-fetched from the "
        "original at full resolution, so you can see the boats you are selecting for. "
        "The even grid is only a starting point — **shift+drag a cell, or nudge it with "
        "the arrow keys**, and it cuts from where you put it; the offsets are saved in "
        "`layout.json` next to the tiles (on every Apply, or with Save) and come back "
        "when you reopen the scene. The output is `split_images/<stem>/` plus the "
        "`tiles.json` manifest, which is what inference reads — and cells cut "
        "earlier show up in blue, with Apply only ever adding to them."
    )
    fields = [
        Field(
            "input",
            "Image folder (-i)",
            kind="choice",
            source="satellite",
            value=rel(SATELLITE_DIR),
            info="The picker lists every image under it.",
        ),
        Field("output", "Tile root (-o, optional)", info="Blank: <image folder>/split_images/"),
        [
            Field("tile", "Cell size (--tile)", value="1024"),
            Field("port", "Port", value=str(DEFAULT_PORT), info=f"The panel below points at {DEFAULT_PORT}."),
        ],
    ]

    def panel(self, options):
        """The form, then the picker itself in an iframe under it."""
        from ..core.ui import render_fields

        inputs = render_fields(self.fields, options)
        gr.HTML(
            f'<iframe id="{FRAME_ID}" src="http://127.0.0.1:{DEFAULT_PORT}/" '
            'style="width:100%;height:78vh;min-height:560px;'
            "border:1px solid var(--border-color-primary);"
            'border-radius:8px;background:#14171c"></iframe>',
            padding=False,
        )
        # The page is served by the job, so it is not there until Start has run.
        reload_button = gr.Button("↻ Reload the picker", size="sm")
        reload_button.click(
            fn=None,
            js=f"() => {{ const f = document.getElementById('{FRAME_ID}'); if (f) f.src = f.src; }}",
        )
        return inputs  # only the fields are wired to Start

    def build(self, params):
        source = folder(params, "input", "image folder")
        output = folder(params, "output", "tile root", required=False)
        tile = positive_int(params, "tile", 1024)
        port = positive_int(params, "port", DEFAULT_PORT)
        if port > 65535:
            raise ValueError(f"port must be <= 65535, got {port}")

        cmd = [
            python_executable(params),
            PICKER_SCRIPT,
            "-i",
            source,
            "--tile",
            str(tile),
            "-p",
            str(port),
        ]
        if output:
            cmd += ["-o", output]

        url = f"http://127.0.0.1:{port}/"
        notes = [f"[webui] the picker is at {url} — hit ↻ under the panel once it is up"]
        meta = {"feature": self.name, "url": url, "outdir": output or f"{source}/split_images", "cmd": " ".join(cmd)}
        return JobSpec(cmd, meta=meta, notes=notes)
