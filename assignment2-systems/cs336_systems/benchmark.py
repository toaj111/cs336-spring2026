from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import get_cosine_lr, AdamW
import argparse
import torch
import timeit
import statistics

def benchmark(
    model: BasicsTransformerLM,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    mode: str,
    warmup_steps: int,
    measure_steps: int,
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

    def run_step(step: int, measure: bool = False):
        times = {}

        if mode != "forward":
            model.zero_grad()

        torch.cuda.synchronize()
        start = timeit.default_timer()

        # Forward + loss
        logits = model(x)
        loss = cross_entropy(logits, targets)

        torch.cuda.synchronize()
        forward_end = timeit.default_timer()
        times["forward"] = forward_end - start

        if mode == "forward":
            times["total"] = times["forward"]
            return times

        # Backward
        loss.backward()

        torch.cuda.synchronize()
        backward_end = timeit.default_timer()
        times["backward"] = backward_end - forward_end

        if mode == "forward-backward-optimizer":
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

    # Warmup
    for step in range(warmup_steps):
        run_step(step)

    torch.cuda.synchronize()

    # Measurement
    measured_time = {
        "forward": [],
        "backward": [],
        "optimizer": [],
        "total": [],
    }

    for step in range(measure_steps):
        times = run_step(warmup_steps + step)

        for key, value in times.items():
            measured_time[key].append(value)

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
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mode", type=str, 
                        choices=["forward", "forward-backward", "forward-backward-optimizer"], 
                        default="forward-backward-optimizer")
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--measure_steps", type=int, default=10)
    args = parser.parse_args()

    model = BasicsTransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta)
    benchmark(model, args.batch_size, args.context_length, args.vocab_size, args.mode, args.warmup_steps, args.measure_steps)
