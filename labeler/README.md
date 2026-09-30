# ship box review

A local web app for reviewing the ship boxes, written because the boxes are 7 px across at the
median and Label Studio makes you zoom in, look, zoom out and pan to the next one, over and over.

Here every box of the image is already magnified, side by side, in the grid on the right, and each
crop is editable where it sits. The grid can be switched from the current image to every box in the
set at once, sorted by box size, filtered to what is still unreviewed. A model's detections can
be loaded beside the annotations, to find the boxes the two disagree on.

## Run

From the repository root:

```powershell
cd C:\Shijie_Li\boat_detection\code\small_boat_detection
python labeler\server.py    # opens http://127.0.0.1:8000 in the browser
```

Other files than the defaults:

```powershell
python labeler\server.py --coco ..\data\annotations\tier_b\val_coco.json --images ..\data\images\all
python labeler\server.py --port 8010 --no-browser
```

With a model's detections beside the annotations (read-only):

```powershell
python labeler\server.py --coco ..\data\annotations\tier_b\all_coco.json --images ..\data\images\all `
                         --pred outputs\<model>\<run>\<test dir>\predictions.json
```

`--pred` takes a COCO results json whose `image_id`s are those of the `--coco` file: the
`predictions.json` that `train.py -c <config> -r <ckpt> --test-only` writes beside the checkpoint,
or the one `tools/inference/torch_inf_dir.py --coco <file>` writes. Detections under
`--pred-min-score` (0.05) are not loaded. The file can also be picked in the page (**O**, field
**Predictions**).

## What it reads and writes

| | |
|---|---|
| reads | `..\data\annotations\tier_b\all_coco.json`, images from `..\data\images\all` |
| writes | `..\data\annotations\tier_b\all_coco_reviewed.json` (COCO, the edited boxes) |
| writes | `..\data\annotations\tier_b\all_coco_reviewed_review.json` (per-box status, deletions, edits) |

These are the defaults, relative to the repository root: the data folder is `code\data`, beside
the repository. `--root` (default `code\`) is how far the file picker in the page may browse.

The input file is never written to. Saving happens 2 seconds after the last change, and on Ctrl+S;
the writes go through a temporary file, so an interrupted save cannot truncate the json. Restarting
the server picks the review back up from what was written. The predictions file is only read.

## Annotations against predictions

A loaded prediction file adds a row of controls to the header: show / hide the detections
(`ship-pred`, cyan, dashed), the score threshold for what is drawn and paired (0.5), and the IoU a
detection needs to be paired with an annotation (0.1). Each image pairs greedily: detections in
falling score order, each to the unpaired annotation it overlaps most, if that IoU reaches the
threshold. The header counts TP / FP / FN from the pairs at IoU ≥ 0.5, and the image list shows per
image how many annotations have no detection (FN) and how many detections have no annotation (FP).

The filter's **Against predictions** group lists:

| | |
|---|---|
| Annotations without a prediction (missed) | annotations no detection pairs with: missed vessels, or boxes that are not vessels |
| Predictions without an annotation (false alarms) | detections with no annotation: false alarms, or vessels the annotation missed |
| Pairs, IoU ranges | pairs by overlap: a low IoU is a detection off target, or an annotation drawn badly |
| All predictions | every detection at or above the score threshold |

An annotation's crop shows its paired detection dashed over it and the IoU in its tag; sort by
**Lowest IoU first** to see the worst pairs first. A detection is never edited: **A** turns the selected
one into an annotation (confirmed), for a vessel the annotation missed.

## The screen

- **left** — the images, with how many of their boxes are reviewed. Green: all done. Amber: partly.
  Red: something is flagged.
- **middle** — the image itself. Wheel zooms at the cursor, right-drag (or Alt+left-drag) pans,
  left-drag on empty space draws a new box, drag a corner of the selected box to resize. Boxes
  under 6 px on screen get a halo so they can be found at all. A drag longer than 12 times its
  width is taken for a slipped pan, not a ship: it turns red while dragged and is not kept.
- **right** — one magnified crop per box. Zoom is per box: each crop is scaled so the box fills
  about 45% of the cell, up to 24x, and the cell says what zoom it used. Drag the box inside the
  crop to move it, drag a corner to resize; the crop stays anchored where it was, so a box you have
  shifted reads as shifted. Click selects and takes the canvas there; double-click zooms the canvas
  in on it.

The edge between a side pane and the canvas can be dragged to make the pane wider or narrower
(the canvas keeps at least 320 px); a double-click on it gives back the default width. The widths
are remembered by the browser.

## Keys

| | |
|---|---|
| arrows | move the selected box 1 px (5 px with Shift) |
| Ctrl + arrows | resize it |
| Space | confirm, and jump to the next unreviewed box |
| X | flag as doubtful |
| D / Delete | delete the box (or the × on its crop) |
| N / P | next / previous box in the grid |
| J / K | next / previous image |
| F | fit the canvas to the image |
| Ctrl+Z / Ctrl+Y | undo / redo |
| Ctrl+S | save now |
| A | the selected detection becomes an annotation |
| O | open the data-source picker |

## A workflow that suits this data

Set the grid to **All images**, the filter to **Unreviewed**, the sort to **Smallest box first**, and walk
the grid with Space. The smallest boxes are where the annotation is worst, and they come first.
Anything you are unsure about gets X, and a second pass with the filter on **Flagged** deals with those
on the canvas, where the full context is visible.
