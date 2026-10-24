import torch
import torch.nn as nn
import os
from typing import IO, BinaryIO

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    states = {}
    states["model"] = model.state_dict()
    states["optimizer"] = optimizer.state_dict()
    states["iter"] = iteration
    torch.save(states, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    states = torch.load(src)
    model.load_state_dict(states["model"])
    optimizer.load_state_dict(states["optimizer"])
    return states["iter"]