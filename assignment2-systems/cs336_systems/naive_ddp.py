import torch
import torch.distributed as dist
import torch.nn as nn

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        for param in module.parameters():
            dist.broadcast(param.data, src=0, async_op=False)
        

    def forward(self, x: torch.Tensor):
        return self.module(x)
