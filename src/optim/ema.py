"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn

from ..core import register
from ..misc import dist_utils


__all__ = ["ModelEMA"]


@register()
class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmups: int = 1000,
        start: int = 0,
    ):
        super().__init__()

        self.module = deepcopy(dist_utils.de_parallel(model)).eval()
        # if next(model.parameters()).device.type != 'cpu':
        #     self.module.half()  # FP16 EMA

        self.decay = decay
        self.warmups = warmups
        self.before_start = 0
        self.start = start
        self.updates = 0  # number of EMA updates
        if warmups == 0:
            self.decay_fn = lambda x: decay
        else:
            self.decay_fn = lambda x: decay * (
                1 - math.exp(-x / warmups)
            )  # decay exponential ramp (to help early epochs)

        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        if self.before_start < self.start:
            self.before_start += 1
            return
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            for ema_vals, model_vals in self._float_groups(dist_utils.de_parallel(model)):
                # Same arithmetic as `v *= d; v += (1 - d) * m` per tensor, as three
                # multi-tensor kernels instead of ~3 launches per state-dict entry.
                scaled = torch._foreach_mul(model_vals, 1 - d)
                torch._foreach_mul_(ema_vals, d)
                torch._foreach_add_(ema_vals, scaled)

    def _float_groups(self, model: nn.Module):
        """Floating-point (ema, model) state tensors, grouped by device and dtype
        for the foreach kernels, cached per model.

        The tensors alias the modules' own storage (state_dict() entries are
        detached views), so they stay valid while the optimizer updates in place;
        the cache is dropped whenever the EMA module is moved or reloaded.
        """
        cache = getattr(self, "_pairs", None)
        if cache is None or cache[0] is not model:
            msd = model.state_dict()
            groups = {}
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    m = msd[k].detach()
                    ema_vals, model_vals = groups.setdefault((v.device, v.dtype, m.device, m.dtype), ([], []))
                    ema_vals.append(v)
                    model_vals.append(m)
            cache = self._pairs = (model, list(groups.values()))
        return cache[1]

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        self._pairs = None
        return self

    def state_dict(self):
        return dict(module=self.module.state_dict(), updates=self.updates)

    def load_state_dict(self, state, strict=True):
        self.module.load_state_dict(state["module"], strict=strict)
        self._pairs = None
        if "updates" in state:
            self.updates = state["updates"]

    def forwad(self):
        raise RuntimeError("ema...")

    def extra_repr(self) -> str:
        return f"decay={self.decay}, warmups={self.warmups}"


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu", use_buffers=True):
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / 2000))

        def ema_avg(avg_model_param, model_param, num_averaged):
            decay = self.decay_fn(num_averaged)
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=use_buffers)
