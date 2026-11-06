from cs336_basics.model import scaled_dot_product_attention
import torch
import statistics
import gc

if __name__ == "__main__":
    device = "cuda:0"
    batch_size = 8
    d_candidates = [16, 32, 64, 128]
    seq_len_candidates = [256, 1024, 4096, 8192, 16384]

    compiled_scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)

    warmup_steps = 10
    measure_steps = 100
    for d in d_candidates:
        for seq_len in seq_len_candidates:
            try:
                forward_peak = []
                backward_peak = []
                before_backward = []

                torch.cuda.synchronize()
                Q = torch.randn(batch_size, seq_len, d, device=device, requires_grad=True)
                K = torch.randn(batch_size, seq_len, d, device=device, requires_grad=True)
                V = torch.randn(batch_size, seq_len, d, device=device, requires_grad=True)

                for _ in range(warmup_steps):
                    y = compiled_scaled_dot_product_attention(Q, K, V)
                    loss = y.sum()
                    if Q.grad is not None:
                        Q.grad.zero_()
                    if K.grad is not None:
                        K.grad.zero_()
                    if V.grad is not None:
                        V.grad.zero_()
                    loss.backward()

                torch.cuda.synchronize()

                for _ in range(measure_steps):
                    torch.cuda.reset_peak_memory_stats()
                    y = compiled_scaled_dot_product_attention(Q, K, V)
                    loss = y.sum()
                    if Q.grad is not None:
                        Q.grad.zero_()
                    if K.grad is not None:
                        K.grad.zero_()
                    if V.grad is not None:
                        V.grad.zero_()
                    torch.cuda.synchronize()
                    forward_peak.append(torch.cuda.max_memory_allocated())
                    before_backward.append(torch.cuda.memory_allocated())
                    torch.cuda.reset_peak_memory_stats()
                    loss.backward()
                    torch.cuda.synchronize()
                    backward_peak.append(torch.cuda.max_memory_allocated())

                mean_forward_peak = statistics.mean(forward_peak)
                mean_before_backward = statistics.mean(before_backward)
                mean_backward_peak = statistics.mean(backward_peak)


                print(f"seq_len={seq_len}, d={d}:", 
                      f"forward_peak={mean_forward_peak / 1024**2:.2f}MiB", 
                      f"before_backward={mean_before_backward / 1024**2:.2f}MiB",
                      f"backward_peak={mean_backward_peak / 1024**2:.2f}MiB",
                      )


            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM: seq_len={seq_len}, d={d}")
                print(e)
                Q.grad = None
                K.grad = None
                V.grad = None
                del Q, K, V, loss, y
                gc.collect()
                torch.cuda.empty_cache()