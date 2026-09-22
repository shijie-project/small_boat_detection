# webui

Gradio dashboard for launching train / test runs, splitting satellite imagery, and the
Label Studio annotation server from a browser.

```bash
gradio webui/app.py      # hot reload: edit any file under webui/ and the page rebuilds
python -m webui          # plain run, from the project root
python webui/webui.py    # same thing
# http://127.0.0.1:8000  (override with WEBUI_HOST / WEBUI_PORT)
```

A plain run never re-reads its source, so refreshing the browser after an edit shows the
old page. **⟳ Restart webui** (under the tabs) fixes that without killing anything: it
drops every `webui` module, imports the app again and swaps the new page into the server
that is already listening, then reloads the browser tab. Running jobs keep going and keep
their consoles, as they do across a hot reload. If the new code fails to import, the old
page stays up and the error shows as a toast, with the traceback in the server's console.
Other browser tabs on the dashboard need a refresh of their own. The button is hidden
under `gradio webui/app.py`, where every save already does this.

Jobs run from the project root, one per **slot**. A slot is what a feature competes with:
train and test share `run` and are serialised, because they both want every GPU; Label
Studio wants none, so it sits in `service` and can stay up across any number of training
runs; the file conversions are CPU work and get `data`. All three can run at once. The
console on the right has one tab per slot. Adding a slot is adding a name to `SLOTS` in
`core/jobs.py` — the layout follows. A feature with `slot = None` starts nothing and gets
no Start / Stop row (Manual split).

A form has one button, Start. Stop lives in the console tab instead, next to the log of
the thing it kills: a slot runs one job at a time, so a Stop per feature was several
buttons for one process, and the tab you started from is not necessarily the tab you are
looking at when you want it to end.

A job started before a hot reload keeps going, keeps streaming, and can still be stopped
(see `_shared_jobs()` in `core/jobs.py`).

## Layout

```
webui/
├── app.py                 the Blocks: one tab per feature, one console per slot
├── core/                  shared by every feature
│   ├── paths.py           project root, config/ckpt/data dirs, safe path resolution
│   ├── discovery.py       what fills the dropdowns
│   ├── jobs.py            the slots, one subprocess each + its log ring buffer
│   ├── restart.py         ⟳ Restart webui: re-import the package, swap the page in place
│   └── ui.py              field rendering, console, start/stop wiring
└── features/              one module per tab
    ├── base.py            Feature/Field/JobSpec + the argument handling train & test share
    ├── train.py
    ├── test.py
    ├── manual_split.py    the cell picker, mounted at /picker/ (tools/dataset/split_picker.py)
    ├── inference.py       detect on a folder of tiles (tools/inference/torch_inf_dir.py)
    ├── label_studio.py    the annotation server, same as `tools/starter.sh label-studio`
    ├── ls_import.py       predictions.json -> annotations in the LS project
    ├── ls_coco.py         an LS export -> the COCO file training reads
    └── split_coco.py      one COCO file + images -> train / val (tools/annotation/random_split_coco.py)
```

**Manual split** is the one tab that is not a form, because choosing which cells of a
122 MP scene to cut cannot be one. `tools/dataset/split_picker.py` is a small web app of
its own — the scene as a cached preview to fly over, plus any single cell cropped from the
original on demand — and the tab is an iframe onto it, so the canvas never goes through
Gradio's event loop. Its routes are one FastAPI app (`make_app`), and the webui mounts it
at `/picker/` on its own server (`Feature.routes()`, collected in `app.py`'s `main()`), so
there is nothing to start: the picker is up whenever the webui is, and
`http://127.0.0.1:8000/picker/` opens it full-window. Every scene under
`../data/satellite_images` is in its list, grouped by folder with how many cells are cut,
and its tiles go to `split_images/<scene>/` in that same folder. Run on its own,
`python tools/dataset/split_picker.py` serves the same app at the root. Apply cuts with
`tools/dataset/tile_satellite.py`'s own code, so the tiles and the `tiles.json` manifest are
exactly what that script would write for the same cells.

A cell can also be moved off its slot (shift+drag, or the arrow keys) so a tile sits over
the harbour rather than across it. It keeps its `r002_c003` name, `tiles.json` records the
position it was really cut from — which is the only thing inference reads — and the offsets
go to `layout.json` beside the tiles, so reopening the scene shows how it was cut.

**Inference** runs on what **Manual split** produced, so its dropdown lists tile folders
rather than images: a `split_images/<scene>/`, or `split_images/` itself for every scene at
once. That second case is the script's `--all`, which the tab always passes when the folder
holds scenes rather than tiles — a job started from a browser has no stdin, and the script
would otherwise stop to ask which scene it meant. It shares the `run` slot with train and
test, since it wants the same GPU.

**LS import** is the step after that: the `predictions.json` an inference run wrote
becomes annotations in the Label Studio project, one per task whose image the file
mentions. It talks to the running server rather than the database, so Label Studio has to
be up — it is the only feature that depends on another one. Dry run is ticked by default,
because this writes into live annotation work; `Undo instead` removes what an earlier
import created (every imported box carries an id starting with `pred`).

**LS → COCO** closes the loop: the *Export → JSON* Label Studio hands back is a
list of tasks with the boxes in percent, and training reads a COCO file with the
boxes in pixels keyed by bare file name. `tools/annotation/ls_to_coco.py` does
that conversion — dropping cancelled annotations, keeping only the newest one per
task, labelling every box category 3 (*ship*), the way `train_coco.json` and
`val_coco.json` already are. The dropdown lists `../data/export/` first, since
that is where Label Studio's own export button writes. *Merge into* adds the
result to an existing dataset rather than starting a new one: ids continue after
that file's, and a task the file already holds is replaced, so re-exporting after
another round of corrections is safe to run twice. Several exports can be picked
at once — the same project across rounds, or several projects — and they are read
in the order listed, so an image two of them share is taken from the later one.

**Train / Val** is the last step before training, and the only one that writes what a
config names directly: `all_coco.json` + `images/all` in, `train_coco.json` +
`images/train` and `val_coco.json` + `images/val` out. *Val share* `0.2` is one
image in five, so train : val is 4 : 1; the draw is stratified on whether an image
has boxes, so the background images (128 of 914 today) land in both splits in the
same proportion rather than piling into one, and *seed* makes it repeatable.
*Images* says how the files get there: `copy` is the safe default, `link`
hard-links them so a second copy of the folder costs no disk, `move` empties the
source, and `none` writes the two json files and leaves the images alone. Running
it again leaves the previous split's files behind unless *Clean* is ticked.

The console repaints on a 1 s `gr.Timer`, so the page always reflects the real job —
reload the browser, open a second tab, or hot reload the server and it picks up again.
Every handler also repaints on the spot, and no button is ever greyed out: clicking Start
while something runs says so, rather than silently doing nothing.

Two hot-reload notes: the browser page belongs to the app that served it, so refresh it
after "Changes detected"; and a job that is running when you reload keeps the `core/jobs.py`
it started with (edits there land once it finishes).

## Adding a feature

1. `features/<name>.py`: subclass `Feature`, declare `fields`, implement
   `build(params) -> JobSpec`, reusing the helpers in `features/base.py`. Raise
   `ValueError` for bad input — it shows up as an error toast.
1. Register it in `features/__init__.py` (`FEATURES`).
1. Set `slot` if it does not belong with the GPU work (the default is `run`). A slot it is
   the first to use gets its own console tab, Stop button included; nothing else to wire.

That is the whole story for a form-shaped feature. Fields render as a text box, a dropdown
(`kind="choice"`, from a discovery `source` or a literal `choices` tuple), or a checkbox
(`kind="flag"`, read back with `flag(params, name)`). For a tab that needs more than fields
(file upload, a gallery of results), override `Feature.panel(options)` and build the
components yourself; return `{name: component}` and the start button wires itself up.
