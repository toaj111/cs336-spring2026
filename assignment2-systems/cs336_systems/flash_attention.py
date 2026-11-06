import triton
import triton.language as tl
import torch
from jaxtyping import Float
from torch import Tensor
from einops import einsum
import math

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, Q: Float[Tensor, "... N_q d"], K: Float[Tensor, "... N_k d"], V: Float[Tensor, "... N_k d"], is_causal=False):
        B = 16 # Q,K,V使用通用的tile size
        device = Q.device
        d = Q.shape[-1]
        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        T_q = (N_q + B - 1) // B # Q的tile数量
        T_k = (N_k + B - 1) // B # K的tile数量
        O: Float[Tensor, "... N_q d"] = torch.empty_like(Q)
        L: Float[Tensor, "... N_q"] = torch.empty(Q.shape[:-1], device=device, dtype=torch.float32) # 使用fp32，保证数值精度
        for i in range(T_q):
            start_q, end_q = B * i, min(B * (i + 1), N_q)
            Q_i: Float[Tensor, "... B_q d"] = Q[..., start_q : end_q, :]
            O_i : Float[Tensor, "... B_q d"] = torch.zeros_like(Q_i)
            m: Float[Tensor, "... B_q"] = torch.full(Q_i.shape[:-1], float("-inf"), device=device, dtype=torch.float32)
            l: Float[Tensor, "... B_q"] = torch.zeros_like(m)
            for j in range(T_k):
                start_k, end_k = B * j, min(B * (j + 1), N_k)
                K_j: Float[Tensor, "... B_k d"] = K[..., start_k : end_k, :]
                V_j: Float[Tensor, "... B_k d"] = V[..., start_k : end_k, :] 
                S_j: Float[Tensor, "... B_q B_k"] = einsum(Q_i, K_j, "... B_q d, ... B_k d -> ... B_q B_k") / math.sqrt(d)
                old_m: Float[Tensor, "... B_q"] = m
                m = torch.maximum(m, S_j.max(dim=-1).values)
                exp_diff: Float[Tensor, "... B_q"] = torch.exp(old_m - m)
                P_j: Float[Tensor, "... B_q B_k"] = torch.exp(S_j - m.unsqueeze(-1))
                l: Float[Tensor, "... B_q"] = exp_diff * l + P_j.sum(-1)
                O_i = exp_diff.unsqueeze(-1) * O_i + einsum(P_j, V_j, "... B_q B_k, ... B_k d -> ... B_q d")

            O_i = l.reciprocal().unsqueeze(-1) * O_i
            O[..., start_q : end_q, :] = O_i
            L[..., start_q : end_q] = m + torch.log(l)
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: Float[Tensor, "... n_queries d"], K: Float[Tensor, "... n_keys d"], V: Float[Tensor, "... n_keys d"], is_causal=False):
        pass


    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError()

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
    ):
        # Program indices
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        # Offset each pointer with the corresponding batch index
        # multiplied with the batch stride for each tensor
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )