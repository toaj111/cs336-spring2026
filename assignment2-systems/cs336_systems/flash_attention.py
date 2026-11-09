import triton
import triton.language as tl
import torch
from jaxtyping import Float, Int, Bool
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
            O_i: Float[Tensor, "... B_q d"] = torch.zeros_like(Q_i)
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
        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, Q: Float[Tensor, "... n_queries d"], K: Float[Tensor, "... n_keys d"], V: Float[Tensor, "... n_keys d"], is_causal=False):
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        batch_size = Q.shape[-3]
        device = Q.device
        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        D = Q.shape[-1]
        O: Float[Tensor, "... N_q d"] = torch.empty_like(Q)
        L: Float[Tensor, "... N_q"] = torch.empty(Q.shape[:-1], device=device, dtype=torch.float32)
        scale = 1.0 / math.sqrt(D)
        Q_TILE_NUM = (N_q + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        flash_fwd_kernel[(Q_TILE_NUM, batch_size,)] (
            Q, K, V,
            O, L,
            Q.stride(-3), Q.stride(-2), Q.stride(-1),
            K.stride(-3), K.stride(-2), K.stride(-1),
            V.stride(-3), V.stride(-2), V.stride(-1),
            O.stride(-3), O.stride(-2), O.stride(-1),
            L.stride(-2), L.stride(-1),
            N_q, N_k,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal,
        )
        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)
        return O

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
    is_causal: tl.constexpr,
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

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0), 
    )        

    dtype_V = V_block_ptr.type.element_ty
    dtype_O = O_block_ptr.type.element_ty
    Q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), tl.float32)
    O_i = tl.zeros(Q_i.shape, tl.float32)
    T_K = tl.cdiv(N_KEYS, K_TILE_SIZE)
    last_k_tile_size = N_KEYS % K_TILE_SIZE
    if is_causal:
        causal_mask_q = tl.arange(query_tile_index * Q_TILE_SIZE, query_tile_index + 1)
    for j in range(T_K):
        K_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_j = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S_j = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            causal_mask_k = tl.arange(j * K_TILE_SIZE, (j + 1) * K_TILE_SIZE)
            causal_mask = causal_mask_q[:, None] >= causal_mask_k[None, :]
        if j == T_K - 1 and last_k_tile_size != 0:
            padding_mask = (tl.arange(0, K_TILE_SIZE) < last_k_tile_size)[None, :]
            if is_causal:
                padding_mask &= causal_mask
            S_j = tl.where(padding_mask, S_j, -1e6) # 使用有限大值，避免出现整个tile的被mask，在old_m - m出现负无穷-负无穷=nan的情况
        elif is_causal:
            S_j = tl.where(causal_mask, S_j, -1e6)
        old_m = m
        m = tl.maximum(m, tl.max(S_j, -1))
        exp_diff = tl.exp(old_m - m)
        P_j = tl.exp(S_j - m[:, None])
        l = exp_diff * l + tl.sum(P_j, -1)
        O_i = exp_diff[:, None] * O_i
        P_j = P_j.to(dtype_V)
        O_i = tl.dot(P_j, V_j, acc=O_i)
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = (1.0 / l[:, None]) * O_i
    O_i = O_i.to(dtype_O)
    l = m + tl.log(l)
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))



