"""The scene page: pick a satellite scene and a model, detect, look, download.

For the people who use the detector rather than train it. Everything the page
does goes through the HTTP API below, so another program can do the same:

    GET  /api/scenes                     the scenes under ../data/satellite_images
    GET  /api/models                     the models of models.yml
    GET  /api/masks?scene=...            the areas to leave out of that scene, saved from the page
    PUT  /api/masks?scene=...            {"polygons": [[[x, y], ...], ...]} in scene px
    POST /api/runs                       {"scene", "model", "threshold", "overlap", "mask"} -> {"run_id"}
    GET  /api/runs                       every run, newest first
    GET  /api/runs/{id}                  status, stage, progress, summary, log tail
    GET  /api/runs/{id}/detections       the vessels, and those the mask set aside, boxes in scene px
    GET  /api/runs/{id}/mask             the mask the run used
    GET  /api/runs/{id}/files/{name}     detections.csv, detections_coco.json, overlay.jpg, ...

with the interactive reference at ``/docs``. The scene viewer's tiles are the
Manual split picker's (``tools/dataset/split_picker.py``), served from the
decoded scene without a pyramid on disk.

    python -m webui.scene        # http://127.0.0.1:8001 (SCENE_HOST / SCENE_PORT)
"""

import os
import sys
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from ..core.paths import PICKER_SCRIPT, ROOT, SATELLITE_DIR
from .runs import Runs, clean_polygons, models, read_mask, write_mask

sys.path.insert(0, str((ROOT / PICKER_SCRIPT).parent))
import split_picker  # noqa: E402

HOST = os.environ.get("SCENE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SCENE_PORT", "8001"))
PAGE = Path(__file__).with_name("static") / "index.html"

library = split_picker.Library(SATELLITE_DIR, 1024)
runs = Runs()

app = FastAPI(
    title="Small boat detection -- scene API",
    description="Detect vessels in a satellite scene: tiling, detection and merging in one run.",
)


def not_found(exc):
    return HTTPException(status_code=404, detail=f"not found: {exc}")


@app.get("/", include_in_schema=False)
def page():
    return HTMLResponse(PAGE.read_bytes())  # read every time: an edit lands on reload


@app.get("/vendor/openseadragon.min.js", include_in_schema=False)
def viewer_js():
    return FileResponse(split_picker.VIEWER_JS, media_type="text/javascript")


@app.get("/api/scenes")
def scenes():
    out = []
    for name in library.sources():
        path = library.path_of(name)
        out.append({"name": name, "stem": path.stem, "mb": round(path.stat().st_size / 2**20, 1)})
    return {"scenes": out}


@app.get("/api/scene", include_in_schema=False)
def scene(name: str):
    try:
        image = library.image(name)
        path = library.path_of(name)
    except KeyError as exc:
        raise not_found(exc) from None
    return {
        "name": name,
        "width": image.width,
        "height": image.height,
        "version": int(path.stat().st_mtime),
        "view_tile": split_picker.VIEW_TILE,
        "top_level": split_picker.max_level(image.width, image.height),
    }


@app.get("/tile", include_in_schema=False)
def tile(name: str, level: int, x: int, y: int):
    try:
        payload = split_picker.tile_jpeg(library, name, level, x, y)
    except KeyError as exc:
        raise not_found(exc) from None
    return Response(payload, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


def scene_name(name):
    try:
        library.path_of(name)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"no such scene: {name!r}") from None
    return name


@app.get("/api/masks")
def get_mask(scene: str):
    return {"scene": scene_name(scene), "polygons": read_mask(scene)}


@app.put("/api/masks")
def put_mask(scene: str, body: dict = Body(...)):  # noqa: B008  (FastAPI's own idiom)
    try:
        polygons = write_mask(scene_name(scene), body.get("polygons", []))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"scene": scene, "polygons": polygons}


@app.get("/api/models")
def list_models():
    return {"models": [{"name": m["name"], "config": m["config"], "checkpoint": m["checkpoint"]} for m in models()]}


@app.post("/api/runs")
def start_run(body: dict = Body(...)):  # noqa: B008  (FastAPI's own idiom)
    try:
        scene_path = library.path_of(body.get("scene", ""))
    except KeyError:
        raise HTTPException(status_code=400, detail=f"no such scene: {body.get('scene')!r}") from None
    available = models()
    wanted = body.get("model") or (available[0]["name"] if available else None)
    model = next((m for m in available if m["name"] == wanted), None)
    if model is None:
        raise HTTPException(status_code=400, detail=f"no such model: {wanted!r}")
    try:
        threshold = float(body.get("threshold", 0.5))
        overlap = int(body.get("overlap", 0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="threshold and overlap must be numbers") from None
    if not 0 <= threshold < 1:
        raise HTTPException(status_code=400, detail="threshold must be in [0, 1)")
    if not 0 <= overlap < 512:
        raise HTTPException(status_code=400, detail="overlap must be in [0, 512) px")
    try:
        polygons = clean_polygons(body.get("mask") or [])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    try:
        run_id = runs.start(scene_path, body["scene"], model, threshold, overlap, polygons)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return {"run_id": run_id}


@app.get("/api/runs")
def list_runs():
    return {"runs": runs.list()}


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    try:
        return runs.status(run_id)
    except KeyError as exc:
        raise not_found(exc) from None


@app.get("/api/runs/{run_id}/detections")
def run_detections(run_id: str):
    try:
        found, aside = runs.detections(run_id)
    except KeyError as exc:
        raise not_found(exc) from None
    return {"detections": found, "masked": aside}


@app.get("/api/runs/{run_id}/mask")
def run_mask(run_id: str):
    try:
        return {"polygons": runs.mask(run_id)}
    except KeyError as exc:
        raise not_found(exc) from None


@app.get("/api/runs/{run_id}/files/{name}")
def run_file(run_id: str, name: str):
    try:
        path = runs.file(run_id, name)
    except KeyError as exc:
        raise not_found(exc) from None
    return FileResponse(path, filename=f"{run_id}_{name}")


def main():
    import uvicorn

    print(f"scene page on http://{HOST}:{PORT}  (API reference at /docs)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
