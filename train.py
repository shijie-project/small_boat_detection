"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys


# Must be set before CUDA initialises. Growable segments stop the allocator from
# fragmenting under this model's per-step changing query counts, i.e. less
# memory reserved-but-unused (Linux only; Windows ignores it with a warning).
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
torch.multiprocessing.set_sharing_strategy("file_system")

import argparse
import time
from pathlib import Path
from pprint import pformat

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils, logger
from src.solver import TASKS


debug = False

if debug:
    import torch

    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def main(args) -> None:
    """main"""
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    # TF32 tensor cores for fp32 matmuls, as convolutions already use by default.
    # (cudnn.benchmark is deliberately left off: its autotuning trials allocate
    # huge workspaces -- peak memory nearly doubled -- and cost ~10 s up front.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    assert not all([args.tuning, args.resume]), "Only support from_scratch or resume or tuning at one time"

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() if k not in ["update"] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    split = "test" if args.test_only else "train"
    if split == "train":
        cfg.output_dir = (Path(cfg.output_dir) / split / time.strftime("%Y%m%d-%H%M%S")).resolve()
    else:
        assert cfg.resume is not None
        cfg.output_dir = (Path(cfg.resume).parent / time.strftime("%Y%m%d-%H%M%S")).resolve()

    cfg.output_dir.mkdir(exist_ok=True, parents=True)

    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if dist_utils.is_main_process():
        logger.tee_print(pformat(cfg.__dict__), file_path=cfg.output_dir.joinpath("log.txt"))

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    # default None, not False: a False default always overrode the config, so
    # `use_amp: True` in the YAML (configs/dome/include/optimizer.yml) never applied
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=None,
        help="auto mixed precision training (default: config's use_amp)",
    )
    parser.add_argument("--output-dir", type=str, help="output directoy")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summry")
    parser.add_argument("--test-only", action="store_true", default=False)

    # priority 1
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    # env
    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")

    parser.add_argument("--local-rank", type=int, help="local rank id")
    args = parser.parse_args()

    main(args)
