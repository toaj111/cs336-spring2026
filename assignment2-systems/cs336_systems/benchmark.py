from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import get_cosine_lr, AdamW
import argparse
import torch
import timeit
import statistics
import torch.cuda.nvtx as nvtx
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor
from cs336_basics.nn_utils import softmax
import math
import cs336_basics

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]

    with nvtx.range("attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

def benchmark(
    model: BasicsTransformerLM,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    mode: str,
    warmup_steps: int,
    measure_steps: int,
    use_amp: bool,
    record_memory: bool,
):
    assert mode in {
        "forward",
        "forward-backward",
        "forward-backward-optimizer",
    }
    assert warmup_steps >= 0
    assert measure_steps >= 2

    model.to("cuda:0")
    model.train()

    x = torch.randint(
        0, vocab_size, (batch_size, context_length), device="cuda:0"
    )
    targets = torch.randint(
        0, vocab_size, (batch_size, context_length), device="cuda:0"
    )

    if mode == "forward-backward-optimizer":
        optimizer = AdamW(model.parameters())

    def run_step(step: int):
        times = {}

        if mode != "forward":
            model.zero_grad()

        torch.cuda.synchronize()
        start = timeit.default_timer()
        with nvtx.range("forward"):
            # Forward + loss
            logits = model(x)
            loss = cross_entropy(logits, targets)

        torch.cuda.synchronize()
        forward_end = timeit.default_timer()
        times["forward"] = forward_end - start

        if mode == "forward":
            times["total"] = times["forward"]
            return times
            
        with nvtx.range("backward"):
            # Backward
            loss.backward()

        torch.cuda.synchronize()
        backward_end = timeit.default_timer()
        times["backward"] = backward_end - forward_end

        if mode == "forward-backward-optimizer":
            with nvtx.range("optimizer"):
                # Optimizer
                lr = get_cosine_lr(
                    step + 1,
                    1e-3,
                    1e-4,
                    0.1 * (warmup_steps + measure_steps),
                    warmup_steps + measure_steps,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.step()

            torch.cuda.synchronize()
            optimizer_end = timeit.default_timer()
            times["optimizer"] = optimizer_end - backward_end
            times["total"] = optimizer_end - start
        else:
            times["total"] = backward_end - start

        return times

    with nvtx.range("warmup"):
        # Warmup
        for step in range(warmup_steps):
            with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
                run_step(step)

    torch.cuda.synchronize()

    # Measurement
    measured_time = {
        "forward": [],
        "backward": [],
        "optimizer": [],
        "total": [],
    }

    if record_memory:
        torch.cuda.memory._record_memory_history(max_entries = 10000)
    with nvtx.range("measure"):
        for step in range(measure_steps):
            with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
                times = run_step(warmup_steps + step)

            for key, value in times.items():
                measured_time[key].append(value)

    if record_memory:
        torch.cuda.memory._dump_snapshot("memeory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    # Print individual measurements
    for i in range(measure_steps):
        print(f"step: {i + 1}")
        for key, values in measured_time.items():
            if values:
                print(f"  {key}: {values[i]:.6f} s")

    # Print statistics
    for key, values in measured_time.items():
        if values:
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            print(f"{key} mean: {mean:.6f} s, std: {std:.6f} s")

    return measured_time
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mode", type=str, 
                        choices=["forward", "forward-backward", "forward-backward-optimizer"], 
                        default="forward-backward")
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--measure_steps", type=int, default=5)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--record_memory", action="store_true", default=False)
    args = parser.parse_args()

    model = BasicsTransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta)
    benchmark(model, args.batch_size, args.context_length, args.vocab_size, args.mode, args.warmup_steps, args.measure_steps, args.use_amp, args.record_memory)
