"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import concurrent.futures
import math
import os
import sys
import time
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from tools.concatenate_images import concatenate_images
from tools.visualize_image_annotation import visualize_detection

from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..misc.logger import tee_print
from ..optim import ModelEMA, Warmup


TRUE_FLAGS = ("true", "1", "yes", "on")

SAVE_INTERMEDIATE_VISUALIZE_RESULT = os.getenv("SAVE_INTERMEDIATE_VISUALIZE_RESULT", "false").lower() in TRUE_FLAGS
SAVE_TEST_VISUALIZE_RESULT = os.environ.get("SAVE_TEST_VISUALIZE_RESULT", "False").lower() in TRUE_FLAGS


def targets_to_device(targets, device):
    """Copy the targets over without blocking (the loader pins them) and drop the
    torchvision tv_tensor subclasses: every op on those goes through a Python
    ``__torch_function__`` hook, and nothing past the transforms needs them."""
    return [{k: v.as_subclass(torch.Tensor).to(device, non_blocking=True) for k, v in t.items()} for t in targets]


def results_to_cpu(results):
    """Postprocessor results -> CPU with one copy per key for the whole batch
    (the evaluator's per-image ``.tolist()`` calls were three syncs per image)."""
    if not results:
        return results
    out = [dict() for _ in results]
    for k in results[0]:
        tensors = [r[k] for r in results]
        if not all(isinstance(t, torch.Tensor) and t.dtype == tensors[0].dtype for t in tensors):
            for o, t in zip(out, tensors):
                o[k] = t.cpu() if isinstance(t, torch.Tensor) else t
            continue
        flat = torch.cat([t.reshape(-1) for t in tensors]).cpu()
        for o, t, part in zip(out, tensors, flat.split([t.numel() for t in tensors])):
            o[k] = part.view(t.shape)
    return out


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs,
):
    log_file = kwargs.get("log_file")
    # Path(None) raises, so keep the fallback on the raw value.
    print_func = print if log_file is None else partial(tee_print, file_path=Path(log_file))

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ", print_func=print_func)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)

    device_type = torch.device(device).type
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets_to_device(targets, device)
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
            for b, target in enumerate(targets):
                image = samples[b].cpu()
                _, H, W = image.shape
                target_cpu = {}
                for k, v in target.items():
                    if k == "boxes":
                        target_cpu[k] = v.cpu().detach().clone() * torch.tensor([W, H, W, H])
                    else:
                        target_cpu[k] = v.cpu().detach().clone()
                visualize_detection(image, target_cpu, "sample_gt", return_image=False, type="xywh")

        if scaler is not None:
            with torch.autocast(device_type=device_type, cache_enabled=True):
                outputs = model(samples, targets=targets)

            if not bool(torch.isfinite(outputs["pred_boxes"]).all()):  # one sync instead of two
                print(outputs["pred_boxes"])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=device_type, enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        # Everything that gets logged comes to the host in one copy; each `.item()`
        # (one per loss term, ~60 of them) used to be its own sync.
        logged = torch.stack([v.detach().float() for v in [loss_value, *loss_dict_reduced.values()]]).tolist()
        loss_value, loss_dict_reduced = logged[0], dict(zip(loss_dict_reduced.keys(), logged[1:]))

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v, global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_func(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    output_dir,
    **kwargs,
):
    log_file = kwargs.get("log_file")
    # Path(None) raises, so keep the fallback on the raw value.
    print_func = print if log_file is None else partial(tee_print, file_path=Path(log_file))

    if SAVE_TEST_VISUALIZE_RESULT:
        visualize_dir = output_dir.joinpath("visualize_results")
        visualize_dir.mkdir(parents=True, exist_ok=True)
        print("Saving visualize results to visualize_dir")

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ", print_func=print_func)
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    # For defe Accuracy calculation
    if model.encoder.use_defe:
        total_defe_samples, ample_defe_predictions, total_anchor_num = 0, 0, 0

    MAX_PENDING_TASKS = 256
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        pending_futures = []

        for samples, targets in metric_logger.log_every(data_loader, 10, header):
            # read on the host before the copy: `.item()` on the device copy is a sync per image
            image_ids = [t["image_id"].item() for t in targets]
            samples = samples.to(device, non_blocking=True)
            targets = targets_to_device(targets, device)

            if SAVE_TEST_VISUALIZE_RESULT:
                coco = data_loader.dataset.coco
                file_names = [coco.loadImgs(id)[0]["file_name"] for id in image_ids]

            outputs = model(samples, targets=None)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            if SAVE_TEST_VISUALIZE_RESULT:
                process_args = []
                scale_factor = float(samples[0].shape[1] / orig_target_sizes[0][0])
                for i in range(len(targets)):
                    sample_cpu = samples[i].cpu()
                    target_cpu = {k: v.cpu() for k, v in targets[i].items()}
                    result_cpu = {k: v.cpu() for k, v in results[i].items()}
                    process_args.append(
                        (
                            sample_cpu,
                            target_cpu,
                            result_cpu,
                            file_names[i],
                            scale_factor,
                            visualize_dir,
                        )
                    )

                if len(pending_futures) >= MAX_PENDING_TASKS:
                    while len(pending_futures) > 0:
                        done_futures = []
                        for future in pending_futures:
                            if future.done():
                                done_futures.append(future)

                        for future in done_futures:
                            pending_futures.remove(future)

                        if not done_futures:
                            time.sleep(0.1)

                for args in process_args:
                    future = executor.submit(process_image_pair, args)
                    pending_futures.append(future)

            res = dict(zip(image_ids, results_to_cpu(results)))
            if coco_evaluator is not None:
                coco_evaluator.update(res)

            if model.encoder.use_defe:
                # For defe Ample Rate calculation
                pred_defe = outputs["batch_queries_num"][0]
                if pred_defe >= targets[0]["labels"].shape[0]:
                    ample_defe_predictions += 1
                total_defe_samples += 1
                total_anchor_num += outputs["batch_queries_num"][0]

        concurrent.futures.wait(pending_futures)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_func(f"Averaged stats: {metric_logger}")

    if model.encoder.use_defe:
        print_func(f"defe Ample Rate: {ample_defe_predictions / total_defe_samples}")
        print_func(f"defe Average Anchor Number: {total_anchor_num / total_defe_samples}")

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize(print_func)

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator


def process_image_pair(args):
    sample, target, result, filename, scale_factor, visualize_dir = args
    visualize_dir = Path(visualize_dir)

    sample_img = visualize_detection(sample, target, f"sample_{filename}", return_image=True)
    result_img = visualize_detection(
        sample,
        result,
        f"result_{filename}",
        scale_factor=scale_factor,
        return_image=True,
    )
    concatenate_images(sample_img, result_img, output_path=visualize_dir.joinpath(filename))
