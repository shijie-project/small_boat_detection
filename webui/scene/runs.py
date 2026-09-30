"""Detection runs: one folder each under ``../data/scene_runs``, one running at a time.

A run is ``tools/inference/detect_scene.py`` in a child process, the same way
the dashboard runs its jobs (``webui/core/jobs.py``): a crash or an out-of-memory
error ends the child, not the page, and the GPU is free again once it exits.

The folder is the record. ``request.json`` is written when the run is started,
``run.json`` by the script when it finishes, and ``log.txt`` once the child has
exited, so a restarted server lists every earlier run as it was. A run given a
mask keeps it as ``mask.json``.

Masks are also kept per scene, in ``../data/scene_masks/<stem>.json``, so the
areas drawn over a scene's buildings are there the next time it is opened.
"""

import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import yaml

from ..core.jobs import Job
from ..core.paths import DATA_DIRS, ROOT

RUNS_ROOT = DATA_DIRS / "scene_runs"
MASKS_ROOT = DATA_DIRS / "scene_masks"
MAX_POLYGONS, MAX_POINTS = 500, 5000
SCRIPT = "tools/inference/detect_scene.py"
MODELS_FILE = Path(__file__).with_name("models.yml")
STAGES = ("tiling", "loading model", "detecting", "merging", "writing", "done")
# the dashboard's training and testing hold 16-24 GB; a desktop holds 2-3
GPU_BUSY_MB = int(os.environ.get("SCENE_GPU_BUSY_MB", "6000"))
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}$")
DOWNLOADS = (
    "detections.csv",
    "detections_coco.json",
    "detections.json",
    "predictions.json",
    "overlay.jpg",
    "mask.json",
    "masked.json",
    "run.json",
    "log.txt",
)


def models():
    """The models of ``models.yml`` whose config and checkpoint are both on disk."""
    with open(MODELS_FILE, encoding="utf-8") as handle:
        entries = yaml.safe_load(handle) or []
    return [entry for entry in entries if (ROOT / entry["config"]).is_file() and (ROOT / entry["checkpoint"]).is_file()]


def gpu_memory_used():
    """MB in use on the first GPU, or ``None`` when nvidia-smi cannot tell."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(out.stdout.split()[0])
    except Exception:  # noqa: BLE001 -- no GPU, no driver: let the run find out
        return None


def clean_polygons(polygons):
    """The polygons as ``[[x, y], ...]`` lists of floats; ``ValueError`` on anything else."""
    if not isinstance(polygons, list) or len(polygons) > MAX_POLYGONS:
        raise ValueError(f"a mask is a list of at most {MAX_POLYGONS} polygons")
    out = []
    for polygon in polygons:
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= MAX_POINTS:
            raise ValueError(f"a polygon has 3 to {MAX_POINTS} points")
        out.append([[round(float(x), 1), round(float(y), 1)] for x, y in polygon])
    return out


def mask_path(scene_name):
    return MASKS_ROOT / f"{Path(scene_name).stem}.json"


def read_mask(scene_name):
    return (read_json(mask_path(scene_name)) or {}).get("polygons", [])


def write_mask(scene_name, polygons):
    MASKS_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"scene": scene_name, "polygons": clean_polygons(polygons)}
    with open(mask_path(scene_name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return payload["polygons"]


def read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


class Runs:
    def __init__(self):
        self.lock = threading.Lock()
        self.job = Job()
        self.current = None  # the run id the job belongs to

    def folder(self, run_id):
        if not RUN_ID_RE.match(run_id or ""):
            raise KeyError(run_id)
        path = RUNS_ROOT / run_id
        if not path.is_dir():
            raise KeyError(run_id)
        return path

    def start(self, scene_path, scene_name, model, threshold, overlap, polygons=()):
        with self.lock:
            if self.job.is_running():
                raise RuntimeError(f"run {self.current} is still going")
            used = gpu_memory_used()
            if used is not None and used > GPU_BUSY_MB:
                raise RuntimeError(
                    f"the GPU already has {used / 1024:.1f} GB in use -- a training or test run? "
                    "Detect again once it has finished."
                )
            self._save_log()  # the previous run's, if its log is still only in memory

            run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            out = RUNS_ROOT / run_id
            out.mkdir(parents=True, exist_ok=False)
            request = {
                "run_id": run_id,
                "started": datetime.now().isoformat(timespec="seconds"),
                "scene": scene_name,
                "model": model["name"],
                "config": model["config"],
                "checkpoint": model["checkpoint"],
                "score_threshold": threshold,
                "overlap": overlap,
                "mask_areas": len(polygons),
            }
            with open(out / "request.json", "w", encoding="utf-8") as handle:
                json.dump(request, handle, indent=1)
            if polygons:
                with open(out / "mask.json", "w", encoding="utf-8") as handle:
                    json.dump({"scene": scene_name, "polygons": polygons}, handle)

            cmd = [
                sys.executable,
                SCRIPT,
                "-i",
                str(scene_path),
                "-c",
                model["config"],
                "-r",
                model["checkpoint"],
                "-o",
                str(out),
                "--thrh",
                f"{threshold:g}",
                "--overlap",
                str(overlap),
            ]
            if polygons:
                cmd += ["--mask", str(out / "mask.json")]
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            ok, message = self.job.start(cmd, env=env, cwd=ROOT)
            if not ok:
                raise RuntimeError(message)
            self.current = run_id
            return run_id

    def _save_log(self):
        """Once the child has exited, its console goes to ``log.txt`` beside the results."""
        if self.current is None or self.job.is_running():
            return
        path = RUNS_ROOT / self.current / "log.txt"
        if not path.exists():
            lines = self.job.status()["lines"]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def status(self, run_id):
        out = self.folder(run_id)
        request = read_json(out / "request.json") or {}
        summary = read_json(out / "run.json")
        live = run_id == self.current
        state = self.job.status() if live else None
        if live and not state["running"]:
            self._save_log()

        if live and state["running"]:
            status = "running"
        elif summary and summary.get("status") == "done":
            status = "done"
        else:
            status = "failed"

        if live:
            lines = state["lines"]
        else:
            log = out / "log.txt"
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
        seen = [line[len("[stage] ") :] for line in lines if line.startswith("[stage] ")]
        progress = detection_progress(lines) if seen and seen[-1] == "detecting" else None
        return {
            "run_id": run_id,
            "status": status,
            "stage": seen[-1] if seen else "starting",
            "stages": STAGES,
            "progress": progress,
            "request": request,
            "summary": summary,
            "log": lines[-40:],
            "files": [name for name in DOWNLOADS if (out / name).is_file()],
        }

    def list(self):
        if not RUNS_ROOT.is_dir():
            return []
        runs = []
        for path in sorted(RUNS_ROOT.iterdir(), reverse=True):
            if not (path.is_dir() and RUN_ID_RE.match(path.name)):
                continue
            request = read_json(path / "request.json") or {}
            summary = read_json(path / "run.json") or {}
            running = path.name == self.current and self.job.is_running()
            runs.append(
                {
                    "run_id": path.name,
                    "scene": request.get("scene"),
                    "model": request.get("model"),
                    "status": "running" if running else ("done" if summary.get("status") == "done" else "failed"),
                    "vessels": summary.get("vessels"),
                }
            )
        return runs

    def detections(self, run_id):
        out = self.folder(run_id)
        payload = read_json(out / "detections_coco.json")
        if payload is None:
            raise KeyError(f"{run_id} has no detections")
        found = [
            {
                "id": ann["id"],
                "bbox": ann["bbox"],  # x, y, w, h in scene px
                "score": ann["score"],
            }
            for ann in payload["annotations"]
        ]
        aside = [
            {"bbox": [x0, y0, round(x1 - x0, 1), round(y1 - y0, 1)], "score": det["score"]}
            for det in read_json(out / "masked.json") or []
            for x0, y0, x1, y1 in [det["bbox_xyxy"]]
        ]
        return found, aside

    def mask(self, run_id):
        return (read_json(self.folder(run_id) / "mask.json") or {}).get("polygons", [])

    def file(self, run_id, name):
        if name not in DOWNLOADS:
            raise KeyError(name)
        path = self.folder(run_id) / name
        if not path.is_file():
            raise KeyError(name)
        return path


TILE_LINE_RE = re.compile(r"^\s+(\d+)/(\d+) tiles$")


def detection_progress(lines):
    """``[done, total]`` from the detector's ``N/M tiles`` lines, or ``None`` before the first."""
    for line in reversed(lines):
        match = TILE_LINE_RE.match(line)
        if match:
            return [int(match.group(1)), int(match.group(2))]
    return None
