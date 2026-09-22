"""What a feature is, plus the argument handling train and test have in common.

A feature declares its form (:class:`Field`) and turns the values back into a
:class:`JobSpec`; the app renders the one and runs the other. Anything a second
feature would also need (config lookup, checkpoint lookup, the torchrun prefix)
lives here rather than in the feature modules.
"""

import sys
import time
from dataclasses import dataclass

from ..core.jobs import RUN_SLOT
from ..core.paths import TRAIN_SCRIPT, resolve


@dataclass
class Field:
    """One control in a feature's form.

    ``kind="choice"`` renders a dropdown, filled either from :mod:`discovery
    <webui.core.discovery>` (``source`` is the key: ``"configs"`` /
    ``"checkpoints"`` / ``"tiles"``) or from a fixed ``choices`` tuple;
    ``kind="multichoice"`` is the same dropdown with several picked at once,
    handing ``build()`` a list instead of a string; ``kind="flag"`` renders a
    checkbox; anything else is a text box. ``value``
    is the default -- for a dropdown, the option to start on -- and ``prefer``
    is a regex fallback for when that option is missing.
    """

    name: str
    label: str
    kind: str = "text"
    source: str = ""
    choices: tuple = ()
    value: str = ""
    prefer: str = ""
    optional: bool = False
    empty_label: str = "(none)"
    info: str = ""


class JobSpec:
    """A command line, the environment tweaks it needs, and what to show about it.

    ``notes`` are lines printed to the console above the job's own output --
    anything the user has to do by hand once it is up.
    """

    def __init__(self, cmd, env=None, meta=None, notes=()):
        self.cmd = cmd
        self.env = dict(env or {})
        self.meta = dict(meta or {})
        self.notes = list(notes)


class Feature:
    """One tab in the dashboard.

    Subclasses set ``name``/``label``/``fields`` and implement :meth:`build`.
    Override :meth:`panel` for a tab that needs more than the declared fields.

    ``slot`` says what the feature competes with (see :mod:`webui.core.jobs`):
    the default puts it in ``run`` with the other GPU work, one at a time.
    ``None`` is a tab that starts nothing, so it gets no Start / Stop row.

    ``wide`` hides the console while the tab is open, giving the panel the whole
    page. A form does not need that; a canvas does.

    :meth:`routes` adds HTTP routes to the webui's own server, for a tab that
    serves a page of its own rather than launching one.
    """

    name = ""
    label = ""
    description = ""
    fields = ()
    slot = RUN_SLOT
    wide = False

    def build(self, params) -> JobSpec:
        raise NotImplementedError

    def routes(self):
        """Starlette routes mounted on the webui's server at launch."""
        return []

    def panel(self, options):
        """Render the tab body; returns ``{field name: component}``."""
        from ..core.ui import render_fields

        return render_fields(self.fields, options)


# --------------------------------------------------------------------------- #
# Form value helpers. They raise ValueError, which the app shows as a toast.
# --------------------------------------------------------------------------- #
def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def text(params, key):
    """One form value as a string -- an untouched dropdown hands back ``None``."""
    value = params.get(key)
    return "" if value is None else str(value).strip()


def flag(params, key):
    """One checkbox as a bool."""
    return bool(params.get(key))


def python_executable(params):
    return text(params, "python") or sys.executable


def whole_int(params, key, default, minimum=1):
    raw = text(params, key) or str(default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value


def positive_int(params, key, default):
    return whole_int(params, key, default, minimum=1)


def existing_file(params, key, kind, required=True):
    """Validate one of the paths the dropdowns were filled with."""
    value = text(params, key)
    if not value:
        if required:
            raise ValueError(f"{kind} is required")
        return ""
    path = resolve(value)
    if path is None or not path.is_file():
        raise ValueError(f"{kind} not found: {value}")
    return value


def existing_files(params, key, kind, required=True):
    """The same for a ``multichoice``: every picked path, in the order picked.

    Duplicates are dropped -- selecting the same export twice would convert it
    twice, and the second pass would only replace the first.
    """
    value = params.get(key)
    raw = value if isinstance(value, (list, tuple)) else [value]
    values = []
    for item in raw:
        item = "" if item is None else str(item).strip()
        if not item or item in values:
            continue
        path = resolve(item)
        if path is None or not path.is_file():
            raise ValueError(f"{kind} not found: {item}")
        values.append(item)
    if not values and required:
        raise ValueError(f"{kind} is required")
    return values


def config_path(params):
    return existing_file(params, "config", "config (-c)")


def gpu_env(params):
    gpus = text(params, "gpus")
    return {"CUDA_VISIBLE_DEVICES": gpus} if gpus else {}


def launcher(python, nproc, port):
    """``train.py`` under torchrun when several GPUs are asked for, else plain python."""
    if nproc > 1:
        return [
            python,
            "-m",
            "torch.distributed.run",
            f"--master_port={port}",
            f"--nproc_per_node={nproc}",
            TRAIN_SCRIPT,
        ]
    return [python, TRAIN_SCRIPT]


def runtime_rows(port):
    """The two rows every ``train.py``-based feature ends with."""
    return [
        [
            Field("gpus", "GPUs (CUDA_VISIBLE_DEVICES)", value="0", info="e.g. 0,1"),
            Field("seed", "Seed", value="42"),
        ],
        [
            Field("nproc", "nproc_per_node", value="1"),
            Field("port", "master_port", value=str(port)),
        ],
    ]
