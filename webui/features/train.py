"""Train: ``train.py -c <config> [-t <ckpt>]``, mirroring scripts/dist_train.sh.

The tuning checkpoint follows the config: picking ``Dome-M-*.yml`` selects
``Dome-M-AITOD-best.pth``, ``Dome-L-*.yml`` selects ``Dome-L-AITOD-best.pth``
(the Dome pretrained weights of that size), and a size with no such checkpoint
falls back to training from scratch. Any other checkpoint can still be picked by
hand; one whose name says a different size than the config is refused.
"""

import dataclasses
import re
from pathlib import Path

from ..core.discovery import list_checkpoints
from .base import (
    Feature,
    Field,
    JobSpec,
    config_path,
    existing_file,
    gpu_env,
    launcher,
    positive_int,
    python_executable,
    runtime_rows,
    timestamp,
)


SIZE_PATTERN = re.compile(r"Dome-([SML])-", re.IGNORECASE)


def model_size(path):
    """``"S"`` / ``"M"`` / ``"L"`` from a Dome config or checkpoint name, else ``None``."""
    match = SIZE_PATTERN.search(Path(str(path or "")).name)
    return match.group(1).upper() if match else None


def pretrained_for(config, checkpoints):
    """The Dome AITOD checkpoint of the config's size, or ``""`` (from scratch)."""
    size = model_size(config)
    if size is None:
        return ""
    wanted = f"Dome-{size}-AITOD-best.pth".lower()
    return next((c for c in checkpoints if Path(c).name.lower() == wanted), "")


class TrainFeature(Feature):
    name = "train"
    label = "Train"
    description = (
        "Start a training run. The tuning checkpoint follows the config: Dome-M → "
        "`Dome-M-AITOD-best.pth`, Dome-L → `Dome-L-AITOD-best.pth` (from scratch when "
        "there is none for that size). Pick another by hand, or *(none)* to train from scratch."
    )
    fields = [
        Field(
            "config",
            "Config (-c)",
            kind="choice",
            source="configs",
            value="configs/dome/Dome-M-AEA.yml",
            prefer="AEA",
        ),
        Field(
            "checkpoint",
            "Tuning checkpoint (-t, optional)",
            kind="choice",
            source="checkpoints",
            optional=True,
            empty_label="(none / from scratch)",
        ),
        *runtime_rows(port=7789),
    ]

    def panel(self, options):
        import gradio as gr

        from ..core.ui import default_of, render_fields

        # start on the pretrained weights of the default config's size
        config_field = self.fields[0]
        start_config = default_of(config_field, list(options.get("configs", [])))
        fields = list(self.fields)
        fields[1] = dataclasses.replace(fields[1], value=pretrained_for(start_config, options.get("checkpoints", [])))
        inputs = render_fields(fields, options)

        def follow(config):
            return gr.update(value=pretrained_for(config, list_checkpoints()))

        inputs["config"].change(follow, inputs=inputs["config"], outputs=inputs["checkpoint"])
        return inputs

    def build(self, params):
        config = config_path(params)
        tuning = existing_file(params, "checkpoint", "tuning checkpoint (-t)", required=False)
        size, tuning_size = model_size(config), model_size(tuning)
        if size and tuning_size and size != tuning_size:
            raise ValueError(
                f"{Path(config).name} is a Dome-{size} model but {Path(tuning).name} is Dome-{tuning_size}: "
                f"pick a Dome-{size} checkpoint, or none to train from scratch"
            )
        seed = positive_int(params, "seed", 42)
        nproc = positive_int(params, "nproc", 1)
        port = positive_int(params, "port", 7789)

        outdir = f"output/{Path(config).stem}/{timestamp()}"
        cmd = launcher(python_executable(params), nproc, port)
        cmd += ["-c", config, "--seed", str(seed), "--output-dir", outdir]
        if tuning:
            cmd += ["-t", tuning]

        meta = {"feature": self.name, "config": config, "outdir": outdir, "cmd": " ".join(cmd)}
        return JobSpec(cmd, env=gpu_env(params), meta=meta)
