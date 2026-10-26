import torch
import torch.nn as nn
import os
from typing import IO, BinaryIO
from cs336_basics.config import Config
from cs336_basics.model import TransformerLM

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes], config: Config | None = None):
    states = {}
    states["model"] = model.state_dict()
    states["optimizer"] = optimizer.state_dict()
    states["iter"] = iteration
    states["config"] = config
    torch.save(states, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: nn.Module | None = None, optimizer: torch.optim.Optimizer | None = None) -> int:
    states = torch.load(src)
    if model is not None:
        model.load_state_dict(states["model"])
    if optimizer is not None:
        optimizer.load_state_dict(states["optimizer"])
    return states["iter"]

def load__model_from_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes]) -> TransformerLM:
    states = torch.load(src)
    config: Config = states["config"]
    model = TransformerLM(config.vocab_size, config.context_length, config.d_model, config.num_layers, config.num_heads, config.d_ff, config.rope_theta, config.device, config.dtype)
    model.load_state_dict(states["model"])
    return model