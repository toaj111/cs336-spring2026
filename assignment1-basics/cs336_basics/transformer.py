import torch.nn as nn
import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from cs336_basics.layers import softmax
import math


class TransformerBlock(nn.Module):
    def __init__(self,   
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                theta: float,
                device: torch.device | None = None,
                dtype: torch.dtype | None = None
                ):
        super().__init__()
        self.rms1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.causal_multihead_self_attention = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = SwiGLUFFN(d_model, device=device, dtype=dtype)
        self.ffn.d_ff = d_ff

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]) -> Float[Tensor, " batch sequence_length d_model"]:
        y = x + self.causal_multihead_self_attention(self.rms1(x))

        y = y + self.ffn(self.rms2(y))

        return y        



class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = 10000, use_rope: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.use_rope = use_rope
        # d_k = d_v
        self.d_k = d_model // num_heads

        if self.use_rope:
            self.RoPE = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)

        std = math.sqrt(1 / d_model)

        self.q_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.q_proj_weight, mean=0, std=std, a=-3*std, b=3*std)

        self.k_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.k_proj_weight, mean=0, std=std, a=-3*std, b=3*std)

        self.v_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.v_proj_weight, mean=0, std=std, a=-3*std, b=3*std)

        self.o_proj_weight = nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.o_proj_weight, mean=0, std=std, a=-3*std, b=3*std)



    def forward(
            self, x: Float[Tensor, " ... seq_len d_model"],
            token_positions: Int[Tensor, "... seq_len"] | None = None
            ) -> Float[Tensor, " ... seq_len d_model"]:

        device = x.device
    
        Q = einsum(x, self.q_proj_weight, "... seq_len d_in, d_out d_in -> ... seq_len d_out")
        K = einsum(x, self.k_proj_weight, "... seq_len d_in, d_out d_in -> ... seq_len d_out")
        V = einsum(x, self.v_proj_weight, "... seq_len d_in, d_out d_in -> ... seq_len d_out")
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=device)
        Q_heads = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k = self.d_k)
        K_heads = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k = self.d_k)
        
        if self.use_rope:
            Q_heads = self.RoPE.forward(Q_heads, token_positions)
            K_heads = self.RoPE.forward(K_heads, token_positions)

        V_heads = rearrange(V, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", d_k = self.d_k)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        attention_output_heads = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)
        attention_output = rearrange(attention_output_heads, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")
        output = einsum(attention_output, self.o_proj_weight, "... seq_len d_in, d_out d_in -> ... seq_len d_out")

        return output


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.g = nn.Parameter(torch.ones(d_model, device=device,dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)

        y = (x / RMS) * self.g

        y = y.to(in_type)
        return y


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.d_ff = round((8 * d_model / 3) / 64) * 64 

        std = math.sqrt(2 / (self.d_model + self.d_ff))

        self.weights1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights1, mean=0, std=std, a=-3*std, b=3*std)

        self.weights2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights2, mean=0, std=std, a=-3*std, b=3*std)

        self.weights3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weights3, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = einsum(x, self.weights1, "... d_model, d_ff d_model -> ... d_ff")
        gate = h1 * torch.sigmoid(h1)
        h3 = einsum(x, self.weights3, "... d_model, d_ff d_model -> ... d_ff")
        h_gated = gate * h3
        y = einsum(h_gated, self.weights2, "... d_ff, d_model d_ff -> ... d_model")

        return y

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        positions = torch.arange(max_seq_len, device=device).unsqueeze(1).to(torch.float32) # (seq_len) -> (seq_len, 1)

        # 论文中维度从1开始，这里从0更简洁
        inv_freq = 1 / theta ** ((torch.arange(d_k // 2, device=device).unsqueeze(0) * 2) / d_k) # (d//2) -> (1, d//2) 
        angles = positions @ inv_freq # (seq_len, d // 2)

        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)
        # 将tensor交给pytorch自动管理
        self.register_buffer("cos_cache", cos_values, persistent=False)
        self.register_buffer("sin_cache", sin_values, persistent=False)



    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        position_cos_values = self.cos_cache[token_positions] # (..., seq_len, d//2)
        position_sin_values = self.sin_cache[token_positions]
        x_even = x[..., 0::2] # (...,seq_len, d//2) 
        x_odd = x[..., 1::2] # (...,seq_len, d//2)

        cos_even = position_cos_values * x_even
        sin_even = position_sin_values * x_even
        cos_odd = position_cos_values * x_odd
        sin_odd = position_sin_values * x_odd

        y_even = cos_even - sin_odd
        y_odd = cos_odd + sin_even

        y = torch.stack([y_even, y_odd], dim=-1) # (..., seq_len, d//2, 2)
        y = y.flatten(start_dim=-2)

        return y

def scaled_dot_product_attention(    
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None
    ) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        masked_attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
    else:
        masked_attention_scores = attention_scores

    attention_weights = softmax(masked_attention_scores, dim=-1)
    output = einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return output

        