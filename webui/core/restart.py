"""Restart: rebuild the dashboard from the code on disk, inside the running server.

A plain run (``python -m webui``, or ``webui/webui.py`` from the IDE) never
re-reads its source, so an edit only shows up once the server is killed and
started again -- which kills every job it is running along with it. This does
what ``gradio webui/app.py``'s reloader does on each save, but on a click: drop
every ``webui`` module (and the ``tools/`` scripts it imports, such as the
Manual split picker), import the app again, and swap the page it builds into
the server that is already listening. The jobs live outside those modules (see
``_shared_jobs()`` in ``jobs.py``), so a run in progress keeps going, keeps its
console, and can still be stopped.
"""

import contextvars
import importlib
import sys
from pathlib import Path

from gradio.utils import BaseReloader

from .paths import ROOT


PACKAGE = "webui"
APP_MODULE = "webui.app"
# Scripts the webui imports rather than launches -- the Manual split picker
# serves from this process -- so their edits have to land on restart too.
TOOLS_DIR = ROOT / "tools"


class _Swap(BaseReloader):
    """Gradio's own hot-reload swap, pointed at a server we already hold."""

    def __init__(self, app):
        self.app = app

    @property
    def running_app(self):
        return self.app


def ours(name, module):
    """The webui's own modules, and anything it imported from ``tools/``."""
    if name == PACKAGE or name.startswith(PACKAGE + "."):
        return True
    file = getattr(module, "__file__", None)
    return name != "__main__" and bool(file) and Path(file).resolve().is_relative_to(TOOLS_DIR)


def reimport():
    """``webui.app`` imported afresh; the old modules come back if that fails."""
    saved = {name: module for name, module in sys.modules.items() if ours(name, module)}
    for name in saved:
        del sys.modules[name]
    try:
        # An empty context: this runs inside a gradio event, and components
        # built there would otherwise attach themselves to the page on screen.
        return contextvars.Context().run(importlib.import_module, APP_MODULE)
    except BaseException:
        for name in [name for name, module in list(sys.modules.items()) if ours(name, module)]:
            del sys.modules[name]
        sys.modules.update(saved)
        raise


def add_routes(app, routes):
    """Mount the routes the server does not have yet, returning their paths.

    ``main()`` mounts every feature's routes at launch, but a server started
    before a feature grew one (the Manual split picker moving in-process) would
    answer that page with a 404 until killed -- which is what this button is
    for not having to do. A route already there is left alone: it looks up the
    current code on every request.
    """
    have = {getattr(route, "path", None) for route in app.router.routes}
    added = []
    for route in routes:
        if route.path not in have:
            app.router.routes.insert(0, route)  # ahead of gradio's, as at launch
            added.append(route.path)
    return added


def restart(running):
    """Swap a freshly built page in for ``running``, the page being served now."""
    app = getattr(running, "server_app", None)
    if app is None:
        raise RuntimeError("this page is not the one being served")
    module = reimport()
    demo = module.demo
    # What main() hands to launch(), which this page never goes through.
    demo.css = module.CONSOLE_CSS
    demo.show_error = running.show_error
    for path in add_routes(app, module.feature_routes()):
        print(f"[webui] mounted {path}/ on the running server", flush=True)
    _Swap(app).swap_blocks(demo)
    demo.config = demo.get_config_file()
    demo.server_app = app  # so the next restart, clicked on this page, finds the server
    return demo
