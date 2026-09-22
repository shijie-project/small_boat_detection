"""Manual split, done on the image itself: pick the 1024 cells worth cutting.

The problem with picking regions inside a form is the image. A 122 MP scene
shown as one preview is 40x too small to judge, and the grid drawn over it is a
mesh of 30 px squares -- you cannot see what you are selecting. So this is not a
form: it is a small web app that serves the scene the way a map is served.

  * the whole scene as one downscaled preview to fly over (cached on disk), and
  * any single cell, cropped from the original at full resolution, on demand --
    so zooming past the preview's resolution shows real pixels, not mush.

Click a cell to take it, drag to sweep a block of them, Apply cuts exactly those
and nothing else. The output is the same as ``tile_satellite.py`` produces --
``split_images/<stem>/`` next to the scene, holding ``<stem>_r000_c000.png``
tiles and a ``tiles.json`` manifest -- so everything downstream (inference, the
annotation import) reads it without knowing a human chose the cells.

The even grid is only where a cell starts. Shift-drag one (or nudge it with the
arrow keys) and it moves off its slot to sit over the harbour rather than across
it; it keeps its ``r002_c003`` name and the manifest records where it actually
came from, which is all anything downstream ever reads. Those offsets are saved
in ``layout.json`` beside the tiles -- written on every Apply and by the Save
button -- so reopening the scene shows how it was cut and lets you carry on.

Cells already on disk from an earlier cut are shown in blue: Apply adds to them,
never deletes, and the manifest is rewritten as the union.

The routes are one FastAPI app (:func:`make_app`). The webui mounts it at
``/picker/`` inside its own server, so the Manual split tab has no Start and no
port; run on its own, this file serves the same app at the root.

Usage
-----
    python tools/dataset/split_picker.py                 # http://127.0.0.1:8011
    python tools/dataset/split_picker.py -p 8011 -i ../data/satellite_images
"""

import argparse
import functools
import io
import json
import os
import re
import sys
import threading
from pathlib import Path

from fastapi import Body, FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image


Image.MAX_IMAGE_PIXELS = None  # these are the images the guard is warning about

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tile_satellite  # noqa: E402  (same folder; the cut has to match it exactly)


IMAGE_SUFFIXES = tile_satellite.IMAGE_SUFFIXES
SPLIT_DIRNAME = tile_satellite.SPLIT_DIRNAME
CACHE_DIRNAME = ".split_picker"  # previews live here, next to the imagery
TILE_NAME_RE = re.compile(r"_r(\d{3})_c(\d{3})\.[a-z]+$", re.IGNORECASE)

HERE = Path(__file__).resolve().parent
PAGE = HERE / "split_picker.html"


# --------------------------------------------------------------------------- #
# The scenes on offer, and the one currently open.
# --------------------------------------------------------------------------- #
class Library:
    """The images that can be cut, and a one-scene cache of decoded pixels.

    Decoding a 122 MP PNG costs seconds and ~370 MB, so exactly one stays
    resident: the scene being looked at. Switching drops the previous one.

    A scene is named by its path under the root (``processed/<scene>.png``), and
    its tiles go to ``split_images/<stem>/`` in the scene's own folder -- unless
    ``out_root`` is given, in which case every scene cuts into that.
    """

    def __init__(self, root, tile, out_root=None):
        self.root = Path(root).resolve()
        self.tile = tile
        self.out_root = Path(out_root).resolve() if out_root else None
        self.lock = threading.Lock()
        self.open_name = None
        self.open_image = None

    def sources(self):
        """Every image under the root, our own output folders left out."""
        found = []
        for base, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in sorted(dirs) if not generated(d)]
            found += [
                str(Path(base, name).relative_to(self.root)).replace("\\", "/")
                for name in sorted(files)
                if name.lower().endswith(IMAGE_SUFFIXES)
            ]
        return found

    def scenes(self):
        """What the scene list shows: each image, its folder, and how many cells are cut."""
        return [
            {
                "name": name,
                "folder": os.path.dirname(name),
                "stem": Path(name).stem,
                "cut": count_cut(self.out_dir(name)),
            }
            for name in self.sources()
        ]

    def path_of(self, name):
        path = (self.root / name).resolve()
        if self.root not in path.parents or not path.is_file():
            raise KeyError(name)
        return path

    def image(self, name):
        """The decoded scene, loading it (and dropping the last one) if needed."""
        with self.lock:
            if self.open_name != name:
                self.open_image = None  # free before allocating the next one
                self.open_image = Image.open(self.path_of(name)).convert("RGB")
                self.open_name = name
            return self.open_image

    def out_dir(self, name):
        stem = Path(name).stem
        if self.out_root is not None:
            return long_path(self.out_root / stem)
        return long_path((self.root / name).parent / SPLIT_DIRNAME / stem)

    def cache_dir(self, name):
        path = self.path_of(name).parent / CACHE_DIRNAME
        path.mkdir(exist_ok=True)
        return path


def generated(dirname):
    """A folder we wrote ourselves -- tiles, crops, detections, caches -- never a source."""
    return dirname in (SPLIT_DIRNAME, CACHE_DIRNAME, "crops") or "_det" in dirname or dirname.startswith(".")


PREFIX = "\\\\?\\"  # what Windows wants before a path longer than 260 characters


def long_path(path):
    """The path in the form Windows accepts past its 260 character limit.

    A scene name here is already 70 characters and appears twice in a tile's
    path (the folder and the file), so a tile root a few levels down reaches
    the limit on its own -- and what comes back is a bare "no such file or
    directory" for a folder that is plainly there. Always prefixing on Windows
    beats guessing where the limit falls; :func:`plain` undoes it for display.
    """
    path = Path(path).resolve()
    if os.name == "nt" and not str(path).startswith(PREFIX):
        return Path(PREFIX + str(path))
    return path


def plain(path):
    """The path as a human should see it, without the Windows long-path prefix."""
    text = str(path)
    return (text[len(PREFIX) :] if text.startswith(PREFIX) else text).replace("\\", "/")


def grid_of(width, height, tile):
    return -(-width // tile), -(-height // tile)  # cols, rows


def count_cut(out_dir):
    """How many tiles a scene has on disk, without opening the scene."""
    if not out_dir.is_dir():
        return 0
    return sum(1 for entry in out_dir.iterdir() if TILE_NAME_RE.search(entry.name))


def cut_cells(out_dir, tile):
    """``[{row, col, x, y}]`` already on disk, at the position they were cut from.

    The manifest is the authority -- a cell that was nudged before it was cut is
    not where its row and column say it is -- and the file names are the fallback
    for a folder cut before any of this existed.
    """
    if not out_dir.is_dir():
        return []
    placed = {}
    manifest_path = out_dir / "tiles.json"
    if manifest_path.is_file():
        with open(manifest_path) as handle:
            for entry in json.load(handle).get("tiles", []):
                placed[entry["name"]] = {
                    "row": entry["row"],
                    "col": entry["col"],
                    "x": entry["x"],
                    "y": entry["y"],
                }

    cells = []
    for entry in out_dir.iterdir():
        match = TILE_NAME_RE.search(entry.name)
        if not (entry.is_file() and match):
            continue
        row, col = int(match.group(1)), int(match.group(2))
        cells.append(placed.get(entry.name, {"row": row, "col": col, "x": col * tile, "y": row * tile}))
    return sorted(cells, key=lambda cell: (cell["row"], cell["col"]))


# --------------------------------------------------------------------------- #
# The layout: where each cell sits, once a hand has moved it.
# --------------------------------------------------------------------------- #
def layout_path(out_dir):
    return out_dir / "layout.json"


def read_layout(out_dir):
    """The saved offsets and unfinished selection for a scene, or empty."""
    path = layout_path(out_dir)
    if not path.is_file():
        return {"cells": [], "picked": []}
    with open(path) as handle:
        saved = json.load(handle)
    return {"cells": saved.get("cells", []), "picked": saved.get("picked", [])}


def write_layout(library, name, layout):
    """Record how this scene is being cut, so the next session picks it up."""
    out_dir = library.out_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = library.image(name)
    cols, rows = grid_of(image.width, image.height, library.tile)
    moved = [cell for cell in layout.get("cells", []) if cell.get("dx") or cell.get("dy")]
    payload = {
        "source": plain(library.path_of(name)),
        "width": image.width,
        "height": image.height,
        "tile": library.tile,
        "cols": cols,
        "rows": rows,
        "cells": moved,  # only the ones that were moved; the rest are on the grid
        "picked": layout.get("picked", []),
    }
    with open(layout_path(out_dir), "w") as handle:
        json.dump(payload, handle, indent=1)
    return {"moved": len(moved), "picked": len(payload["picked"]), "path": plain(layout_path(out_dir))}


# --------------------------------------------------------------------------- #
# Preview and cell rendering.
# --------------------------------------------------------------------------- #
def preview_jpeg(library, name, max_side):
    """The whole scene as one JPEG, at most ``max_side`` on its longest edge.

    Cached beside the imagery: re-opening a scene should be instant, and the
    decode is the expensive part of the whole app.
    """
    path = library.path_of(name)
    cache = library.cache_dir(name) / f"{path.stem}_p{max_side}.jpg"
    if cache.is_file() and cache.stat().st_mtime >= path.stat().st_mtime:
        return cache.read_bytes()

    image = library.image(name).copy()
    image.thumbnail((max_side, max_side), Image.LANCZOS, reducing_gap=3.0)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85, optimize=True)
    cache.write_bytes(buffer.getvalue())
    return buffer.getvalue()


def crop_cell(image, x, y, tile):
    """The tile-sized canvas at ``(x, y)``: real content top-left, black beyond.

    Position is given in source pixels rather than derived from a row and column
    -- a cell that has been nudged is no longer where its grid slot is.
    """
    crop = image.crop((x, y, min(x + tile, image.width), min(y + tile, image.height)))
    content = crop.size
    if content != (tile, tile):
        canvas = Image.new("RGB", (tile, tile), (0, 0, 0))
        canvas.paste(crop, (0, 0))
        crop = canvas
    return crop, content


def cell_jpeg(library, name, x, y, size):
    """One cell, cropped from the original at ``(x, y)`` and scaled to ``size`` px."""
    image = library.image(name)
    if x >= image.width or y >= image.height or x < 0 or y < 0:
        raise KeyError(f"cell at ({x}, {y}) is outside the image")

    crop, _ = crop_cell(image, x, y, library.tile)
    if size != library.tile:
        crop = crop.resize((size, size), Image.LANCZOS if size < library.tile else Image.NEAREST)

    buffer = io.BytesIO()
    crop.save(buffer, "JPEG", quality=82)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Applying the selection: the same cut tile_satellite.py makes.
# --------------------------------------------------------------------------- #
def write_cells(library, name, cells, image_format, quality):
    """Cut the chosen cells into ``split_images/<stem>/``; returns a report."""
    path = library.path_of(name)
    image = library.image(name)
    tile = library.tile
    out_dir = library.out_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "." + image_format.lower().lstrip(".")

    written = []
    for cell in cells:
        row, col = int(cell["row"]), int(cell["col"])
        # where the cell actually sits, which is its grid slot only until it is moved
        x = int(cell.get("x", col * tile))
        y = int(cell.get("y", row * tile))
        if not (0 <= x < image.width and 0 <= y < image.height):
            continue
        crop, (content_w, content_h) = crop_cell(image, x, y, tile)

        window = {"region": None, "row": row, "col": col}
        tile_file = tile_satellite.tile_name(path.stem, window, suffix)
        if suffix in (".jpg", ".jpeg"):
            crop.save(out_dir / tile_file, "JPEG", quality=quality)
        else:
            crop.save(out_dir / tile_file, "PNG", compress_level=1)

        written.append(
            {
                "name": tile_file,
                "region": None,
                "row": row,
                "col": col,
                "x": x,
                "y": y,
                "width": tile,
                "height": tile,
                "content_w": content_w,
                "content_h": content_h,
            }
        )

    kept = merge_manifest(library, name, written)
    return {"written": len(written), "total": len(kept), "out_dir": plain(out_dir)}


def merge_manifest(library, name, written):
    """Rewrite ``tiles.json`` as every tile now in the folder, old ones included.

    A manual cut adds to whatever was cut before -- a tile already imported into
    Label Studio must not lose its entry just because this run did not produce
    it -- so the manifest is the union, minus anything whose file has gone.
    """
    path = library.path_of(name)
    image = library.image(name)
    tile = library.tile
    out_dir = library.out_dir(name)
    manifest_path = out_dir / "tiles.json"

    entries = {}
    if manifest_path.is_file():
        with open(manifest_path) as handle:
            for entry in json.load(handle).get("tiles", []):
                entries[entry["name"]] = entry
    for entry in written:
        entries[entry["name"]] = entry

    # a tile file with no entry (hand-copied, or an older naming) still counts
    for cell in cut_cells(out_dir, tile):
        row, col = cell["row"], cell["col"]
        if any(e["row"] == row and e["col"] == col for e in entries.values()):
            continue
        x, y = cell["x"], cell["y"]
        tile_file = f"{path.stem}_r{row:03d}_c{col:03d}.png"
        entries[tile_file] = {
            "name": tile_file,
            "region": None,
            "row": row,
            "col": col,
            "x": x,
            "y": y,
            "width": tile,
            "height": tile,
            "content_w": min(tile, image.width - x),
            "content_h": min(tile, image.height - y),
        }

    live = [entry for entry in entries.values() if (out_dir / entry["name"]).is_file()]
    live.sort(key=lambda entry: (entry["row"], entry["col"]))
    cols, rows = grid_of(image.width, image.height, tile)
    manifest = {
        "source": plain(path),
        "width": image.width,
        "height": image.height,
        "tile": tile,
        "overlap": 0,
        "stride": tile,
        "mode": "manual",
        "cols": cols,
        "rows": rows,
        "padded_width": cols * tile,
        "padded_height": rows * tile,
        "skipped_blank": 0,
        "skipped_black": 0,
        "tiles": live,
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=1)
    return live


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def scene_info(library, name):
    image = library.image(name)
    cols, rows = grid_of(image.width, image.height, library.tile)
    out_dir = library.out_dir(name)
    return {
        "name": name,
        "stem": Path(name).stem,
        "width": image.width,
        "height": image.height,
        "tile": library.tile,
        "cols": cols,
        "rows": rows,
        "cut": cut_cells(out_dir, library.tile),
        "layout": read_layout(out_dir),
        "out_dir": plain(out_dir),
    }


def guarded(route):
    """Errors come back as ``{"error": ...}``, which is what the page shows.

    Caught per route rather than by an app-wide handler: mounted in the webui,
    an exception that escaped this app would end up in gradio's error handling.
    """

    @functools.wraps(route)
    def wrapper(*args, **kwargs):
        try:
            return route(*args, **kwargs)
        except KeyError as exc:
            return JSONResponse({"error": f"not found: {exc}"}, status_code=404)
        except Exception as exc:  # a broken scene must not take the server down
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return wrapper


def make_app(library):
    """The picker -- page, scene data and the cut -- relative to wherever it is mounted.

    The page asks for ``api/scenes``, ``cell?...`` and so on without a leading
    slash, so the same app works at ``/`` on its own and at ``/picker/`` in the
    webui.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    def page():
        return HTMLResponse(PAGE.read_bytes())  # read every time: an edit lands on reload

    @app.get("/api/scenes")
    @guarded
    def scenes():
        return {"scenes": library.scenes(), "tile": library.tile, "root": plain(library.root)}

    @app.get("/api/scene")
    @guarded
    def scene(name: str):
        return scene_info(library, name)

    @app.get("/preview")
    @guarded
    def preview(name: str, max_side: int = Query(4096, alias="max")):
        return Response(preview_jpeg(library, name, max_side), media_type="image/jpeg")

    @app.get("/cell")
    @guarded
    def cell(name: str, x: int, y: int, size: int = 512):
        payload = cell_jpeg(library, name, x, y, max(128, min(size, library.tile)))
        return Response(payload, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.post("/api/layout")
    @guarded
    def layout(body: dict = Body(...)):
        report = write_layout(library, body["name"], body.get("layout", {}))
        print(f"[picker] layout: {report['moved']} moved, {report['picked']} picked -> {report['path']}", flush=True)
        return report

    @app.post("/api/apply")
    @guarded
    def apply(body: dict = Body(...)):
        name = body["name"]
        cells = body.get("cells", [])
        if not cells:
            return JSONResponse({"error": "nothing selected"}, status_code=400)
        report = write_cells(library, name, cells, body.get("format", "png"), int(body.get("quality", 95)))
        print(
            f"[picker] {name}: wrote {report['written']} tile(s) -> {report['out_dir']} ({report['total']} in total)"
        )
        # the layout is saved with every cut: how a scene was cut is part of the
        # result, not something to remember to press a button for
        if body.get("layout") is not None:
            write_layout(library, name, body["layout"])
        report["cut"] = cut_cells(library.out_dir(name), library.tile)
        return report

    return app


def main(args):
    import uvicorn

    if not PAGE.is_file():
        raise SystemExit(f"missing the page: {PAGE}")
    library = Library(args.input, args.tile, args.output)
    if not library.root.is_dir():
        raise SystemExit(f"no such folder: {library.root}")

    print(f"{len(library.sources())} image(s) under {library.root}")
    print(f"tiles go to {library.out_root or '<scene folder>/' + SPLIT_DIRNAME}/<stem>/")
    print(f"manual split picker on http://{args.host}:{args.port}")
    uvicorn.run(make_app(library), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", default="../data/satellite_images", help="folder of source images")
    parser.add_argument(
        "-o", "--output", default=None, help=f"tile root (default: {SPLIT_DIRNAME}/ beside each scene)"
    )
    parser.add_argument("--tile", type=int, default=1024, help="cell size in px (default 1024)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=8011)
    main(parser.parse_args())
