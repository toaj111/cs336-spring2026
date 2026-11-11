
import gc
import csv
import math
import torch
import triton
from itertools import product

from cs336_systems.flash_attention import (
    flash_fwd_kernel,
    torch_flash_attention_backward,
)


def pytorch_attention(Q, K, V):
    d = Q.shape[-1]
    N = Q.shape[-2]

    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)

    mask = torch.triu(
        torch.ones(N, N, device=Q.device, dtype=torch.bool),
        diagonal=1,
    )

    S = S.masked_fill(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)

    return torch.matmul(P, V)


def benchmark(fn):
    # Return milliseconds.
    return triton.testing.do_bench(
        fn,
        warmup=100,
        rep=300,
        return_mode="median",
    )


def run_case(seq_len, d, dtype, device):
    Q = torch.randn(
        1, seq_len, d, device=device, dtype=dtype,
        requires_grad=True,
    )
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn_like(Q, requires_grad=True)

    dO = torch.randn_like(Q)

    results = {}

    # --------------------------------------------------
    # PyTorch baseline
    # --------------------------------------------------

    O_pt = pytorch_attention(Q, K, V)

    def pt_forward():
        return pytorch_attention(Q, K, V)

    def pt_backward():
        return torch.autograd.grad(
            O_pt, (Q, K, V),
            grad_outputs=dO,
            retain_graph=True,
        )

    def pt_end_to_end():
        O = pytorch_attention(Q, K, V)
        return torch.autograd.grad(
            O, (Q, K, V),
            grad_outputs=dO,
        )

    results["pt_forward"] = benchmark(pt_forward)
    results["pt_backward"] = benchmark(pt_backward)
    results["pt_e2e"] = benchmark(pt_end_to_end)

    del O_pt

    # --------------------------------------------------
    # Triton implementation
    # --------------------------------------------------

    def triton_forward(Q, K, V, q_tile_size=16, k_tile_size=32):
        B, N, D = Q.shape

        O = torch.empty_like(Q)
        L = torch.empty((B, N), device=Q.device, dtype=torch.float32)

        grid = (triton.cdiv(N, q_tile_size), B)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *O.stride(),
            *L.stride(),
            N, N,
            D ** -0.5,
            D,
            q_tile_size,
            k_tile_size,
            True,  # is_causal
        )

        return O, L

    O_tri, L_tri = triton_forward(Q, K, V)

    def tri_forward():
        return triton_forward(Q, K, V)
    def tri_backward():
        return torch_flash_attention_backward(
            Q, K, V, O_tri, dO, L_tri
        )

    def tri_end_to_end():
        O, L = triton_forward(Q, K, V)
        return torch_flash_attention_backward(
            Q, K, V, O, dO, L
        )

    results["tri_forward"] = benchmark(tri_forward)
    results["tri_backward"] = benchmark(tri_backward)
    results["tri_e2e"] = benchmark(tri_end_to_end)

    del Q, K, V, dO, O_tri, L_tri
    gc.collect()
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    device = "cuda:0"

    assert torch.cuda.is_available()
    print(torch.cuda.get_device_name(device))


    dtypes = [torch.bfloat16, torch.float32]

    results = []

    for seq_len, d, dtype in product(
        [2**i for i in range(7, 17)],
        [2**i for i in range(4, 8)],
        dtypes,
    ):
        print(
            f"Benchmarking N={seq_len}, D={d}, "
            f"dtype={dtype}"
        )

        try:
            result = run_case(seq_len, d, dtype, device)

            results.append({
                "seq_len": seq_len,
                "d": d,
                "dtype": str(dtype),
                **result,
            })

        except torch.cuda.OutOfMemoryError:
            print("OOM: skipping this configuration")
            torch.cuda.empty_cache()
            gc.collect()

    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)
