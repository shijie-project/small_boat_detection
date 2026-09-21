"""Restart: rebuild the dashboard from the code on disk, inside the running server.

A plain run (``python -m webui``, or ``webui/webui.py`` from the IDE) never
re-reads its source, so an edit only shows up once the server is killed and
started again -- which kills every job it is running along with it. This does
what ``gradio webui/app.py``'s reloader does on each save, but on a click: drop
every ``webui`` module, import the app again, and swap the page it builds into
the server that is already listening. The jobs live outside those modules (see
``_shared_jobs()`` in ``jobs.py``), so a run in progress keeps going, keeps its
console, and can still be stopped.
"""

import contextvars
import importlib
import sys

from gradio.utils import BaseReloader


PACKAGE = "webui"
APP_MODULE = "webui.app"


class _Swap(BaseReloader):
    """Gradio's own hot-reload swap, pointed at a server we already hold."""

    def __init__(self, app):
        self.app = app

    @property
    def running_app(self):
        return self.app


def ours(name):
    return name == PACKAGE or name.startswith(PACKAGE + ".")


def reimport():
    """``webui.app`` imported afresh; the old modules come back if that fails."""
    saved = {name: module for name, module in sys.modules.items() if ours(name)}
    for name in saved:
        del sys.modules[name]
    try:
        # An empty context: this runs inside a gradio event, and components
        # built there would otherwise attach themselves to the page on screen.
        return contextvars.Context().run(importlib.import_module, APP_MODULE)
    except BaseException:
        for name in [name for name in sys.modules if ours(name)]:
            del sys.modules[name]
        sys.modules.update(saved)
        raise


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
    _Swap(app).swap_blocks(demo)
    demo.config = demo.get_config_file()
    demo.server_app = app  # so the next restart, clicked on this page, finds the server
    return demo
