"""The dashboard itself: one tab per feature, one shared console.

Hot reload -- edit any file under ``webui/`` and the page rebuilds itself:

    gradio webui/app.py

Plain run (no reload on save; the ⟳ Restart webui button reloads on demand):

    python -m webui
"""

import sys
from pathlib import Path


# gradio's reloader runs this file as a script, so the project has to be
# importable before anything below can be.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from webui.core.discovery import options
from webui.core.jobs import SLOT_LABELS, SLOTS
from webui.core.paths import HOST, PORT, ROOT
from webui.core.ui import (
    CONSOLE_CSS,
    flatten,
    make_clear,
    make_refresh,
    make_rescan,
    make_restart,
    make_start,
    make_stop,
)
from webui.features import all_features


TITLE = "Small Boat Detection"


def make_console_visibility(feature):
    """Tab handler: the console hides for a ``wide`` feature and comes back after."""

    def visibility():
        return gr.update(visible=not feature.wide)

    return visibility


def slots_of(features):
    """The slots in use, in :data:`SLOTS` order; anything unlisted comes last."""
    used = list(dict.fromkeys(feature.slot for feature in features))
    return [s for s in SLOTS if s in used] + [s for s in used if s not in SLOTS]


def build_ui():
    opts = options()
    features = all_features()
    pending_starts = []  # wired once the console components exist
    choice_fields = []
    choice_components = []
    consoles = {}  # slot -> [status, log, seen]
    slots = slots_of(features)

    with gr.Blocks(title=TITLE, fill_width=True) as demo:
        gr.Markdown(f"### {TITLE} — train / test / annotate")

        tabs = []  # (tab, feature), wired once the console column exists
        with gr.Row():
            # 5/6 rather than 4/7: nine tab labels have to fit across the top of
            # this column, and Gradio hides the ones that do not behind a "...".
            with gr.Column(scale=5, min_width=380):
                with gr.Tabs():
                    for feature in features:
                        with gr.Tab(feature.label) as tab:
                            tabs.append((tab, feature))
                            if feature.description:
                                gr.Markdown(feature.description)
                            inputs = feature.panel(opts)
                            names = list(inputs)
                            # Start is the only button a form needs: stopping
                            # belongs next to the log that shows what is running.
                            button = gr.Button(f"Start {feature.label}", variant="primary")
                            pending_starts.append((button, feature, names, [inputs[n] for n in names]))
                            for field in flatten(feature.fields):
                                if field.kind in ("choice", "multichoice"):
                                    choice_fields.append(field)
                                    choice_components.append(inputs[field.name])

                with gr.Row():
                    rescan_button = gr.Button("↻ Rescan", variant="secondary")
                    # Under `gradio webui/app.py` every save already does this.
                    restart_button = gr.Button("⟳ Restart webui", variant="secondary", visible=not demo.dev_mode)
                gr.Markdown(f"<sub>root: `{ROOT}`</sub>")

            with gr.Column(scale=6) as console_column:
                with gr.Tabs():
                    for slot in slots:
                        with gr.Tab(SLOT_LABELS.get(slot, slot)):
                            with gr.Row():
                                status = gr.Markdown("⚪ **idle**")
                                stop_button = gr.Button("Stop", variant="stop", size="sm", scale=0, min_width=90)
                                clear_button = gr.Button("Clear console", size="sm", scale=0, min_width=140)
                            log = gr.Textbox(
                                label="log",
                                lines=30,
                                max_lines=30,
                                autoscroll=True,
                                interactive=False,
                                show_label=False,
                                buttons=["copy"],
                                elem_classes="console",
                            )
                            consoles[slot] = [status, log, gr.State(())]
                            stop_button.click(make_stop(slot), outputs=consoles[slot])
                            clear_button.click(make_clear(slot), outputs=consoles[slot])

        for button, feature, names, components in pending_starts:
            console = consoles[feature.slot]
            button.click(make_start(feature, names), inputs=components, outputs=console)
        rescan_button.click(make_rescan(choice_fields), outputs=choice_components)
        # The server now serves the rebuilt page; this one is stale, so fetch it.
        restart_button.click(make_restart(demo)).success(fn=None, js="() => { location.reload(); }")
        # A wide feature takes the whole page: the console it would share the
        # row with is worth less than the pixels while you are working in it.
        for tab, feature in tabs:
            tab.select(make_console_visibility(feature), outputs=console_column)
        for slot, console in consoles.items():
            gr.Timer(1.0).tick(
                make_refresh(slot),
                inputs=[console[2]],
                outputs=console,
                show_progress="hidden",
            )

    return demo


demo = build_ui()


def main():
    demo.launch(server_name=HOST, server_port=PORT, css=CONSOLE_CSS, show_error=True)


if __name__ == "__main__":
    main()
