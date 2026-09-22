"""Runs one child process per slot and keeps its output in a ring buffer.

Training and testing both want every GPU on the box, so the dashboard
deliberately serialises them: starting a job while another is running in the
same slot is refused rather than queued. A slot is that mutual exclusion --
features that compete for the same machine share one, features that do not get
their own. Long-lived servers (Label Studio) sit in ``service`` so that leaving
one up does not block the GPU work in ``run``.
"""

import os
import signal
import subprocess
import sys
import threading
import time
import types
from collections import deque
from datetime import datetime

from .paths import ROOT


MAX_LINES = 5000
STATE_MODULE = "_webui_job_state"

RUN_SLOT = "run"  # one at a time: they fight over the GPUs
SERVICE_SLOT = "service"  # stays up until stopped
DATA_SLOT = "data"  # CPU-bound data prep; no reason to wait for a GPU
SLOTS = (RUN_SLOT, SERVICE_SLOT, DATA_SLOT)
SLOT_LABELS = {
    RUN_SLOT: "Train / test",
    SERVICE_SLOT: "Services",
    DATA_SLOT: "Data prep",
}


def _popen_kwargs():
    """Line-buffered text pipes, plus whatever it takes to kill the tree later."""
    kwargs = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        encoding="utf-8",
        errors="replace",
    )
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["preexec_fn"] = os.setsid
    return kwargs


class Job:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._lines = deque(maxlen=MAX_LINES)  # (seq, text)
        self._seq = 0
        self._meta = {}
        self._run = 0  # bumped per start, so a lingering reader can be told apart

    # -- state ------------------------------------------------------------ #
    def is_running(self):
        proc = self._proc
        return proc is not None and proc.poll() is None

    def status(self, after=0):
        """Log lines after ``after`` plus everything the console shows."""
        proc = self._proc
        running = proc is not None and proc.poll() is None
        with self._lock:
            lines = [text for (seq, text) in self._lines if seq > after]
            cursor = self._seq
            size = len(self._lines)
            meta = dict(self._meta)
        if not running and proc is not None and meta.get("exit_code") is None:
            # Don't wait for the reader thread: a child that inherited the pipe
            # (dataloader workers, torchrun ranks) can keep it open long after
            # the process we launched is gone.
            meta["exit_code"] = proc.returncode
        return {"running": running, "meta": meta, "lines": lines, "cursor": cursor, "size": size}

    def _emit(self, text):
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, text))

    def clear(self):
        """Empty the console. ``size`` changes, so the page repaints."""
        with self._lock:
            self._lines.clear()

    # -- lifecycle -------------------------------------------------------- #
    def start(self, cmd, env=None, meta=None, cwd=ROOT, notes=()):
        if self.is_running():
            return False, "A job is already running. Stop it first."

        with self._lock:
            self._run += 1
            run = self._run
            self._lines.clear()
            self._seq = 0
            self._meta = dict(meta or {})
            self._meta["started"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._meta["exit_code"] = None
        self._emit("$ " + " ".join(cmd))
        for note in notes:
            self._emit(note)
        self._emit("")

        try:
            self._proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, **_popen_kwargs())
        except Exception as exc:  # noqa: BLE001 -- surfaced in the log instead
            self._proc = None
            self._emit(f"[webui] failed to launch: {exc}")
            return False, str(exc)

        with self._lock:
            self._meta["pid"] = self._proc.pid

        threading.Thread(target=self._pump, args=(self._proc, run), daemon=True).start()
        return True, "started"

    def _pump(self, proc, run):
        def emit(text):
            if self._run == run:  # a later job owns the console now; stay quiet
                self._emit(text)

        try:
            for line in iter(proc.stdout.readline, ""):
                emit(line.rstrip("\n"))
        except Exception as exc:  # noqa: BLE001
            emit(f"[webui] reader error: {exc}")
        finally:
            proc.wait()
            with self._lock:
                if self._run == run:
                    self._meta["exit_code"] = proc.returncode
            emit("")
            emit(f"[webui] process exited with code {proc.returncode}")

    def stop(self):
        if not self.is_running():
            return False, "No running job."
        proc = self._proc
        try:
            if os.name == "nt":
                # Kill the whole tree; torchrun spawns worker children.
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(2)
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception as exc:  # noqa: BLE001
            self._emit(f"[webui] stop error: {exc}")
            return False, str(exc)
        self._emit("[webui] stop requested by user")
        return True, "stopping"


def _shared_jobs():
    """One job per slot, carried across gradio's hot reload.

    Reloading drops every module under ``webui/`` from ``sys.modules``, so a
    plain module-level singleton would come back empty and we would lose the
    handle to a run that is still going -- no log, no way to stop it. A
    synthetic module has no ``__file__``, so the file watcher leaves it alone.

    Only a *running* job is carried over: it keeps the code it was started
    with, so edits to this file land as soon as it finishes. When nothing is
    running, every reload gets a fresh instance of the freshly loaded class.
    """
    state = sys.modules.get(STATE_MODULE)
    if state is None:
        state = types.ModuleType(STATE_MODULE)
        sys.modules[STATE_MODULE] = state
    jobs = getattr(state, "jobs", None)
    if jobs is None:
        # Reloading over a pre-slot server: adopt its job rather than orphan it.
        legacy = getattr(state, "job", None)
        jobs = {RUN_SLOT: legacy} if legacy is not None and legacy.is_running() else {}
        state.jobs = jobs
    for slot in SLOTS:
        if slot not in jobs or not jobs[slot].is_running():
            jobs[slot] = Job()
    return jobs


JOBS = _shared_jobs()


def job(slot=RUN_SLOT):
    """The job of one slot. A slot we have not seen gets one, never a KeyError."""
    existing = JOBS.get(slot)
    if existing is None:
        existing = JOBS[slot] = Job()
    return existing
