"""Manual split: ``tools/dataset/split_picker.py``, served by the webui itself.

Everything else here is a form, because everything else *is* a form. Choosing
which 1024 cells of a 122 MP scene to cut is not: it needs the image at real
resolution, zoomed and panned, with the grid drawn on it. So the picker is its
own little web app -- a canvas that never goes through Gradio's event loop --
and this tab is a window onto it.

It used to be a job: a second server on a port of its own, started from a form,
with an iframe that showed nothing until Start had run and ↻ was clicked. Now
the webui mounts the picker's routes at ``/picker/`` on its own server, so the
tab has no Start, no port and no console, and the picker is there the moment the
tab opens. Scenes are chosen inside it, grouped by folder.
"""

import importlib
import sys
import threading

import gradio as gr
from starlette.routing import Mount

from ..core.paths import PICKER_SCRIPT, ROOT, SATELLITE_DIR
from .base import Feature


MOUNT = "/picker"
TILE = 1024  # the network's input size; the CLI's --tile is there for anything else
FRAME_ID = "manual-split-frame"

_lock = threading.Lock()
_app = None


def picker_app():
    """The picker's FastAPI app over the satellite folder, built on first use.

    Imported late: the picker pulls in cv2 through ``tile_satellite``, and the
    rest of the dashboard should not wait for that or fail with it.
    """
    global _app
    with _lock:
        if _app is None:
            folder = str((ROOT / PICKER_SCRIPT).parent)
            if folder not in sys.path:
                sys.path.insert(0, folder)
            import split_picker

            _app = split_picker.make_app(split_picker.Library(SATELLITE_DIR, TILE))
        return _app


async def dispatch(scope, receive, send):
    """Hand the request to this module's current picker.

    Looked up on every request rather than bound once: ⟳ Restart webui imports
    this module afresh, and the route registered at launch has to follow it.
    """
    module = importlib.import_module(__name__)
    await module.picker_app()(scope, receive, send)


class ManualSplitFeature(Feature):
    name = "manual-split"
    label = "Manual split"
    slot = None  # nothing to start: the picker is part of the server
    wide = True  # the canvas wants the page, not a third of it
    description = (
        "Pick the cells worth cutting and **Apply** cuts them into `split_images/<scene>/` beside the "
        "scene; picks and moves are saved as you go. The keys are in the corner of the picker. "
        f'<a href="{MOUNT}/" target="_blank">Open in a new window ↗</a>'
    )

    def routes(self):
        return [Mount(MOUNT, app=dispatch)]

    def panel(self, options):
        gr.HTML(
            f'<iframe id="{FRAME_ID}" src="{MOUNT}/" allow="fullscreen" '
            'style="width:100%;height:calc(100vh - 210px);min-height:560px;'
            "border:1px solid var(--border-color-primary);"
            'border-radius:8px;background:#14171c"></iframe>',
            padding=False,
        )
        return {}
