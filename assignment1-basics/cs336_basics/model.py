import torch.nn as nn
import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.layers import softmax, Linear, Embedding
import math
from cs336_basics.transformer import TransformerBlock, RMSNorm

class TransformerLM(nn.Module):
    def __init__(self, 
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None
                ):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
            for _ in range(num_layers)
        ])
        self.rms = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, " batch_size sequence_length"]):
        y = self.embedding(x)
        for block in self.transformer_blocks:
            y = block.forward(y)

        y = self.rms(y)
        y = self.linear(y)

        return y