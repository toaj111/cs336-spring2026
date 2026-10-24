from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import torch.nn as nn


# TODO: 应用lr strategy
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0 or weight_decay < 0 or eps < 0 or betas[0] < 0 or betas[1] < 0:
            raise ValueError("Invalid hyperparameter")
        defaults = {"alpha": lr, "lam": weight_decay, "eps": eps, "beta1": betas[0], "beta2": betas[1]}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            lam = group["lam"]
            eps = group["eps"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros(p.shape))
                v = state.get("v", torch.zeros(p.shape))
                alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= alpha * lam * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                p.data -= alpha_t * m / (v.sqrt() + eps)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss



class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


def cosine_lr_schedule_with_warmup(t: int, alpha_max: float, alpha_min: float, t_w: int, t_c: int) -> float:
    if t < t_w:
        return t / t_w * alpha_max
    elif t <= t_c:
        return alpha_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm) -> None:
    params = [p for p in parameters if p.grad is not None]

    if not params:
        return

    norm = torch.sqrt(sum(torch.sum(p.grad.detach() ** 2) for p in params))
    if norm < max_l2_norm:
        return
    clip_coef = max_l2_norm / (norm + 1e-6)
    for p in params:
        p.grad.detach().mul_(clip_coef)



