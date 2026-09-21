"""The Gradio pieces every feature tab shares: forms, console, start/stop wiring.

Feature modules stay declarative -- a list of :class:`~webui.features.base.Field`
and a ``build()`` -- and this module turns that into components hooked onto the
shared job runner.
"""

import os
import re
from contextlib import nullcontext

import gradio as gr

from .discovery import options
from .jobs import job
from .paths import ROOT


LOG_TAIL = 400  # lines kept on screen; the job's own buffer holds far more

CONSOLE_CSS = """
.console textarea {
  font-family: "Cascadia Code", Consolas, monospace !important;
  font-size: 12.5px !important;
  line-height: 1.45 !important;
  white-space: pre !important;
}
"""


# --------------------------------------------------------------------------- #
# Forms
# --------------------------------------------------------------------------- #
def choices_of(field, opts):
    """Dropdown choices for ``field``, blank option first when it is optional.

    A multi-select has no blank option: picking nothing is how it says nothing.
    """
    values = list(field.choices) if field.choices else list(opts.get(field.source, []))
    blank = [(field.empty_label, "")] if field.optional and field.kind != "multichoice" else []
    return blank + [(value, value) for value in values], values


def default_of(field, values):
    """The exact default when it is available, else the first ``prefer`` match."""
    if field.value and field.value in values:
        return field.value
    if field.prefer:
        pattern = re.compile(field.prefer, re.IGNORECASE)
        for value in values:
            if pattern.search(value):
                return value
    return "" if field.optional else None


def component_of(field, opts):
    if field.kind == "flag":
        return gr.Checkbox(value=bool(field.value), label=field.label, info=field.info or None)
    if field.kind not in ("choice", "multichoice"):
        return gr.Textbox(value=field.value, label=field.label, info=field.info or None)
    choices, values = choices_of(field, opts)
    chosen = default_of(field, values)
    if field.kind == "multichoice":
        # The form hands back a list; ``existing_files`` is the helper that reads it.
        return gr.Dropdown(
            choices=choices,
            value=[chosen] if chosen else [],
            multiselect=True,
            label=field.label,
            info=field.info or None,
        )
    return gr.Dropdown(
        choices=choices,
        value=chosen,
        label=field.label,
        info=field.info or None,
    )


def flatten(fields):
    """Fields in declaration order, with the row groupings unwrapped."""
    for entry in fields:
        yield from entry if isinstance(entry, (list, tuple)) else [entry]


def render_fields(fields, opts):
    """``{name: component}`` for one feature; a nested list becomes a row."""
    inputs = {}
    for entry in fields:
        group = list(entry) if isinstance(entry, (list, tuple)) else [entry]
        with gr.Row() if len(group) > 1 else nullcontext():
            for field in group:
                inputs[field.name] = component_of(field, opts)
    return inputs


# --------------------------------------------------------------------------- #
# Console
# --------------------------------------------------------------------------- #
def status_markdown(state):
    meta = state["meta"]
    feature = meta.get("feature", "")
    if state["running"]:
        head = f"🟢 **running** — {feature}"
    elif meta.get("exit_code") is not None:
        code = meta["exit_code"]
        head = f"{'✅' if code == 0 else '❌'} **exited ({code})** — {feature}"
    else:
        return "⚪ **idle**"
    parts = [head]
    if state["running"] and meta.get("pid"):
        parts.append(f"pid `{meta['pid']}`")
    if meta.get("started"):
        parts.append(f"started `{meta['started']}`")
    if state["running"] and meta.get("url"):
        parts.append(f"[{meta['url']}]({meta['url']})")
    if meta.get("outdir"):
        parts.append(f"output `{meta['outdir']}`")
    return " · ".join(parts)


def paint(slot):
    """One slot's console outputs: ``[status, log, change key]``.

    Every handler returns this, so a click lands immediately instead of waiting
    for the next tick -- a button that takes a second to react reads as broken.
    """
    state = job(slot).status()
    # .get: a job carried across a hot reload answers with its old code.
    key = (state["cursor"], state.get("size", 0), state["running"], state["meta"].get("exit_code"))
    return [status_markdown(state), "\n".join(state["lines"][-LOG_TAIL:]), key]


def make_refresh(slot):
    """Timer handler: repaint the console, or skip when nothing moved."""

    def refresh(seen):
        outputs = paint(slot)
        return [gr.skip()] * 3 if outputs[-1] == seen else outputs

    return refresh


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
# Buttons stay clickable at all times: a greyed-out button that does nothing
# when the job it is waiting on never exits (torchrun hanging on shutdown is
# the usual one) is indistinguishable from a broken page. Every click either
# does something or says why it did not.
def make_start(feature, names):
    """Click handler: form values in, running job out, console repainted."""

    def start(*values):
        params = dict(zip(names, values))
        try:
            spec = feature.build(params)
        except ValueError as exc:
            raise gr.Error(str(exc)) from None

        env = os.environ.copy()
        env.update(spec.env)
        env["PYTHONUNBUFFERED"] = "1"

        ok, message = job(feature.slot).start(spec.cmd, env=env, meta=spec.meta, cwd=ROOT, notes=spec.notes)
        if not ok:
            raise gr.Error(message)
        gr.Info(f"{feature.label} started")
        return paint(feature.slot)

    return start


def make_stop(slot):
    def stop():
        ok, message = job(slot).stop()
        (gr.Info if ok else gr.Warning)(message)
        return paint(slot)

    return stop


def make_clear(slot):
    def clear():
        job(slot).clear()
        return paint(slot)

    return clear


def make_rescan(fields):
    """Re-read configs / checkpoints without restarting the server."""

    def rescan():
        opts = options()
        updates = [gr.update(choices=choices_of(field, opts)[0]) for field in fields]
        gr.Info(
            f"{len(opts['configs'])} configs · {len(opts['checkpoints'])} checkpoints"
            f" · {len(opts['satellite'])} image folders · {len(opts['tiles'])} tile folders"
        )
        return updates[0] if len(updates) == 1 else updates

    return rescan
