import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import statistics

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, data_size):
    setup(rank, world_size)
    # warmup stage
    torch.cuda.synchronize()
    for _ in range(5):
        data = torch.randint(0, 10, (data_size,), device="cuda:0")
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()

    # measure stage
    measured_times = []
    torch.cuda.synchronize()
    for _ in range(5):
        data = torch.randint(0, 10, (data_size,), device="cuda:0")
        torch.cuda.synchronize()
        start = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()
        elapsed_time = timeit.default_timer() - start
        measured_times.append(elapsed_time)

    mean_time = statistics.mean(measured_times)
    print(f"world_size={world_size}, data_size={4*data_size//1000000}MB, mean_time={mean_time}")
    

if __name__ == "__main__":
    num_processes = [2, 4, 6]
    data_sizes = [250000, 2500000, 25000000, 250000000]
    for world_size in num_processes:
        for data_size in data_sizes:
            mp.spawn(fn=distributed_demo, args=(world_size, data_size, ), nprocs=world_size, join=True)