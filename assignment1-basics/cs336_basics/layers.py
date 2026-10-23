import torch.nn as nn
import torch
from einops import einsum
import math



class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weights, mean=0, std=std, a=-3*std, b=3*std)

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
        return y    


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]



def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    probs = exp_x / sum_exp
    return probs



