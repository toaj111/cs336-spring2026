import torch
from cs336_basics.layers import softmax
from jaxtyping import Float, Int
from torch import Tensor

def cross_entropy(logits: Float[Tensor, "... batch_size vocab_size"], targets: Int[Tensor, "... batch_size"]) -> Float[Tensor, ""]:
    max_o = logits.max(dim=-1).values # (b, )
    target_o = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) # (b, )
    log_sum_exp = torch.exp(logits - max_o.unsqueeze(-1)).sum(dim=-1).log() # (b, )
    loss = log_sum_exp - (target_o - max_o)
    avg_loss = loss.mean()
    return avg_loss
    