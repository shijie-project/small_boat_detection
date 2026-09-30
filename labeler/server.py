"""
A small review GUI for the ship boxes: a local web app, started with

    python server.py

and opened at http://127.0.0.1:8000. The boxes here are 7 px across at the median, so the point of
the thing is the grid of zoomed crops on the right: every box of the image (or of the whole set)
side by side, each one magnified, editable where it sits.

Reads the COCO file, writes a copy: the annotations to ``--out`` and the review status
(confirmed / flagged / deleted / edited) to a sidecar beside it. The input file is never touched.

``--pred`` adds a model's detections for the same images, read-only: a COCO results json (the
``predictions.json`` that ``train.py --test-only`` or ``torch_inf_dir.py`` writes, image ids from the
same COCO file), or a COCO file whose annotations carry a ``score``. The page draws them beside the
annotations as ``ship-pred``, pairs them with the annotations by IoU, and can list the annotations
no detection found, the detections no annotation explains, and the pairs by how well they overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import webbrowser
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).parent
# the repository sits beside the data: code/small_boat_detection/labeler -> code/data
DEFAULT_DATA = (HERE / ".." / ".." / "data").resolve()


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class Store:
    """The COCO file, its review sidecar and the atomic writes back to disk."""

    def __init__(
        self,
        coco_path: Path,
        images_dir: Path,
        out_path: Path | None,
        root: Path,
        pred_path: Path | None = None,
        pred_min_score: float = 0.05,
    ):
        self.lock = threading.Lock()
        self.root = root
        self.pred_min_score = pred_min_score
        self.load(coco_path, images_dir, out_path, pred_path)

    def load(
        self, coco_path: Path, images_dir: Path, out_path: Path | None = None, pred_path: Path | None = None
    ) -> None:
        """Point the store at another dataset; ``--out`` defaults to a sibling of the input."""
        coco_path = coco_path.resolve()
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        # read the predictions before anything is replaced, so a bad file leaves the old dataset open
        preds, pred_info = self._read_predictions(pred_path, {im["id"] for im in coco["images"]})
        self.coco_path = coco_path
        self.images_dir = images_dir.resolve()
        self.out_path = (out_path or coco_path.with_name(coco_path.stem + "_reviewed.json")).resolve()
        self.review_path = self.out_path.with_name(self.out_path.stem + "_review.json")
        self.coco = coco
        self.preds, self.pred_info = preds, pred_info
        # a resumed session reads back what was written last, so the review survives a restart
        self.review = (
            json.loads(self.review_path.read_text(encoding="utf-8"))
            if self.review_path.exists()
            else {"status": {}, "deleted": [], "edited": []}
        )
        if self.out_path.exists():
            saved = json.loads(self.out_path.read_text(encoding="utf-8"))
            self.coco["annotations"] = saved.get("annotations", self.coco["annotations"])

    def _read_predictions(self, path: Path | None, image_ids: set) -> tuple[list[dict], dict]:
        """The detections, as ``{id, image_id, bbox, score}`` with negative ids so they can never be
        mistaken for an annotation. Those under ``pred_min_score`` are left out: a DETR keeps its
        100-300 queries per image, almost all near zero, and the page has no use for them."""
        if path is None:
            return [], {}
        path = path.resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("annotations", []) if isinstance(raw, dict) else raw
        if not isinstance(rows, list) or (rows and not all("bbox" in r and "image_id" in r for r in rows)):
            raise ValueError("expected a COCO results list, or a COCO file with scored annotations")
        preds, unknown, below = [], 0, 0
        for r in rows:
            score = float(r.get("score", 1.0))
            if score < self.pred_min_score:
                below += 1
                continue
            if r["image_id"] not in image_ids:
                unknown += 1
                continue
            preds.append(
                {
                    "id": -(len(preds) + 1),
                    "image_id": r["image_id"],
                    "bbox": [round(float(v), 2) for v in r["bbox"]],
                    "score": round(score, 4),
                }
            )
        info = {
            "path": str(path),
            "n_read": len(rows),
            "n_kept": len(preds),
            "n_below_min_score": below,
            "n_unknown_image": unknown,
            "min_score": self.pred_min_score,
        }
        return preds, info

    def missing_images(self, limit: int = 5) -> list[str]:
        """The first few image files the COCO file names but the chosen folder does not hold."""
        missing = []
        for im in self.coco["images"]:
            if not (self.images_dir / im["file_name"]).is_file():
                missing.append(im["file_name"])
                if len(missing) >= limit:
                    break
        return missing

    def state(self) -> dict[str, Any]:
        """Everything the page needs: 914 images and 1688 boxes fit in one response."""
        by_image: dict[int, int] = {}
        for a in self.coco["annotations"]:
            by_image[a["image_id"]] = by_image.get(a["image_id"], 0) + 1
        images = [
            {
                "id": im["id"],
                "file_name": im["file_name"],
                "width": im["width"],
                "height": im["height"],
                "n_boxes": by_image.get(im["id"], 0),
            }
            for im in self.coco["images"]
        ]
        return {
            "categories": self.coco["categories"],
            "images": images,
            "annotations": self.coco["annotations"],
            "predictions": self.preds,
            "pred": self.pred_info,
            "review": self.review,
            "paths": {
                "coco": str(self.coco_path),
                "out": str(self.out_path),
                "review": str(self.review_path),
                "images": str(self.images_dir),
                "root": str(self.root),
                "pred": self.pred_info.get("path", ""),
            },
            "missing_images": self.missing_images(),
        }

    def save(self, annotations: list[dict], review: dict) -> dict[str, Any]:
        with self.lock:
            for a in annotations:  # keep COCO's own fields consistent with the edited box
                x, y, w, h = (round(float(v), 2) for v in a["bbox"])
                a["bbox"] = [x, y, w, h]
                a["area"] = round(w * h, 2)
                a.setdefault("segmentation", [])
                a.setdefault("iscrowd", 0)
                a.setdefault("ignore", 0)
            self.coco["annotations"] = annotations
            self.review = review
            out = dict(self.coco)
            out["info"] = {**self.coco.get("info", {}), "description": f"reviewed from {self.coco_path.name}"}
            _write_atomic(self.out_path, out)
            _write_atomic(self.review_path, review)
        return {"ok": True, "n_annotations": len(annotations), "out": str(self.out_path)}


def _write_atomic(path: Path, obj: Any) -> None:
    """Write through a temporary file, so a crash mid-save cannot leave a half-written json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def build_app(store: Store) -> FastAPI:
    app = FastAPI(title="boat labeler")

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return store.state()

    @app.post("/api/save")
    def post_save(payload: dict[str, Any]) -> dict[str, Any]:
        return store.save(payload["annotations"], payload["review"])

    @app.get("/images/{name}")
    def get_image(name: str) -> FileResponse:
        path = (store.images_dir / name).resolve()
        if store.images_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(404, "no such image")
        return FileResponse(path)

    @app.get("/api/browse")
    def browse(path: str | None = None) -> dict[str, Any]:
        """One directory: its subdirectories, its json files, and how many images it holds."""
        here = _inside_root(Path(path) if path else store.root, store.root)
        if not here.is_dir():
            raise HTTPException(400, f"not a directory: {here}")
        dirs, files, n_images = [], [], 0
        for e in sorted(here.iterdir(), key=lambda e: e.name.lower()):
            try:
                if e.is_dir():
                    dirs.append({"name": e.name, "path": str(e)})
                elif e.suffix.lower() == ".json":
                    files.append({"name": e.name, "path": str(e), "size": e.stat().st_size})
                elif e.suffix.lower() in IMAGE_SUFFIXES:
                    n_images += 1
            except OSError:
                continue  # a locked or vanished entry is simply not offered
        parent = here.parent if here != here.parent and store.root in here.parents else None
        return {
            "path": str(here),
            "parent": str(parent) if parent else None,
            "root": str(store.root),
            "dirs": dirs,
            "files": files,
            "n_images": n_images,
        }

    @app.post("/api/open")
    def open_dataset(payload: dict[str, Any]) -> dict[str, Any]:
        """Switch to another COCO file / image folder. The page reloads its state afterwards."""
        coco = _inside_root(Path(payload["coco"]), store.root)
        images = _inside_root(Path(payload["images"]), store.root)
        out = _inside_root(Path(payload["out"]), store.root) if payload.get("out") else None
        pred = _inside_root(Path(payload["pred"]), store.root) if payload.get("pred") else None
        if out and out.suffix.lower() != ".json":
            raise HTTPException(400, f"the output must be a .json file: {out.name}")
        if out and out in (coco, pred):
            raise HTTPException(400, "the output would overwrite the file being read")
        if not coco.is_file():
            raise HTTPException(400, f"no such file: {coco}")
        if not images.is_dir():
            raise HTTPException(400, f"no such folder: {images}")
        if pred and not pred.is_file():
            raise HTTPException(400, f"no such file: {pred}")
        try:
            with store.lock:
                store.load(coco, images, out, pred)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            raise HTTPException(400, f"cannot read {coco.name}{' / ' + pred.name if pred else ''}: {e}") from e
        return store.state()

    app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
    return app


def _inside_root(path: Path, root: Path) -> Path:
    """Browsing and opening stay under --root, so a stray path cannot walk the whole disk."""
    p = path.resolve()
    if p != root and root not in p.parents:
        raise HTTPException(403, f"outside --root ({root}): {p}")
    return p


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", type=Path, default=DEFAULT_DATA / "annotations" / "tier_b" / "all_coco.json")
    p.add_argument("--images", type=Path, default=DEFAULT_DATA / "images" / "all")
    p.add_argument("--out", type=Path, default=None, help="default: <coco>_reviewed.json beside the input")
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATA.parent,
        help="the folder the in-page file picker may browse (default: code/)",
    )
    p.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="a model's detections on the same images (COCO results json), shown read-only",
    )
    p.add_argument(
        "--pred-min-score", type=float, default=0.05, help="detections below this score are not loaded (default 0.05)"
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    store = Store(
        args.coco.resolve(), args.images.resolve(), args.out, args.root.resolve(), args.pred, args.pred_min_score
    )
    print(f"  boxes : {len(store.coco['annotations'])} in {len(store.coco['images'])} images")
    print(f"  read  : {store.coco_path}")
    if store.pred_info:
        info = store.pred_info
        print(
            f"  pred  : {info['path']}: {info['n_kept']} of {info['n_read']} detections kept "
            f"(score >= {info['min_score']}), {info['n_unknown_image']} on images not in the COCO file"
        )
    print(f"  images: {store.images_dir}")
    print(f"  write : {store.out_path}")
    print(f"  review: {store.review_path}")
    print(f"  root  : {store.root}  (what the picker in the page can browse)")
    if miss := store.missing_images():
        print(f"  ! {len(miss)}+ images named in the COCO file are not in that folder, e.g. {miss[0]}")
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  open  : {url}\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(build_app(store), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
