import numpy.typing as npt
import torch
from jaxtyping import Int
from torch import Tensor
import random

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
                 ) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    N = len(dataset)
    if N - context_length - 1 < 0:
        raise ValueError("Context length is too big!")
    
    inputs = torch.zeros(batch_size, context_length, device=device, dtype=torch.int)
    targets = torch.zeros(batch_size, context_length, device=device, dtype=torch.int)
    for i in range(batch_size):
        s = random.randint(0, N - context_length - 1)
        inputs[i] = torch.tensor(dataset[s:s+context_length])
        targets[i] = torch.tensor(dataset[s+1:s+context_length+1])

    return (inputs, targets)
