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
    used = list(dict.fromkeys(feature.slot for feature in features if feature.slot))
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
        with gr.Row(equal_height=True):
            gr.Markdown(f"### {TITLE} — train / test / annotate", scale=1)
            rescan_button = gr.Button("↻ Rescan", variant="secondary", size="sm", scale=0, min_width=110)
            # Under `gradio webui/app.py` every save already does this.
            restart_button = gr.Button(
                "⟳ Restart webui", variant="secondary", size="sm", scale=0, min_width=150, visible=not demo.dev_mode
            )

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
                            # Every tab starts and stops its own job -- if it has one.
                            if feature.slot:
                                with gr.Row():
                                    button = gr.Button(f"Start {feature.label}", variant="primary")
                                    stop_button = gr.Button(f"Stop {feature.label}", variant="stop")
                                pending_starts.append(
                                    (button, stop_button, feature, names, [inputs[n] for n in names])
                                )
                            for field in flatten(feature.fields):
                                if field.kind in ("choice", "multichoice"):
                                    choice_fields.append(field)
                                    choice_components.append(inputs[field.name])

                gr.Markdown(f"<sub>root: `{ROOT}`</sub>")

            with gr.Column(scale=6) as console_column:
                with gr.Tabs():
                    for slot in slots:
                        with gr.Tab(SLOT_LABELS.get(slot, slot)):
                            with gr.Row():
                                status = gr.Markdown("⚪ **idle**")
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
                            clear_button.click(make_clear(slot), outputs=consoles[slot])

        labels = {feature.name: feature.label for feature in features}
        for button, stop_button, feature, names, components in pending_starts:
            console = consoles[feature.slot]
            button.click(make_start(feature, names), inputs=components, outputs=console)
            stop_button.click(make_stop(feature, labels), outputs=console)
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


def feature_routes():
    """Pages the features serve themselves (the Manual split picker), on this same server."""
    return [route for feature in all_features() for route in feature.routes()]


def main():
    # ahead of gradio's own routes; a tab that gains one later gets it on restart
    demo.launch(
        server_name=HOST,
        server_port=PORT,
        css=CONSOLE_CSS,
        show_error=True,
        app_kwargs={"routes": feature_routes()},
    )


if __name__ == "__main__":
    main()
