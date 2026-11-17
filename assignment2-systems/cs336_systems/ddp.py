import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from typing import Type, Any, Callable

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        # for param in module.parameters():
        #     dist.broadcast(param.data, src=0, async_op=False)
        param_list = [param for param in module.parameters()]
        flattened_params = _flatten_dense_tensors(param_list)
        dist.broadcast(flattened_params, src=0, async_op=False)
        unflattend_params = _unflatten_dense_tensors(flattened_params, param_list)
        for orig, new in zip(param_list, unflattend_params):
            orig.data.copy_(new)
        

    def forward(self, x: torch.Tensor):
        return self.module(x)

class OverlapDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        param_list = [param for param in module.parameters()]
        flattened_params = _flatten_dense_tensors(param_list)
        dist.broadcast(flattened_params, src=0, async_op=False)
        unflattend_params = _unflatten_dense_tensors(flattened_params, param_list)
        for orig, new in zip(param_list, unflattend_params):
            orig.data.copy_(new)

        for param in module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._backward_hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def _backward_hook(self, tensor: torch.Tensor):
        handle = dist.all_reduce(tensor.grad, dist.ReduceOp.AVG, async_op=True)
        self.handles.append(handle)

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        super().__init__(params, kwargs)
        self.optimizer = optimizer_cls(self.param_groups[0]["sharded_params"], **kwargs)

    def step(self, closure: Callable | None = None, **kwargs):
        loss = self.optimizer.step(closure)
        handles = []
        world_size = dist.get_world_size()
        for group in self.param_groups:
            n = len(group["params"])
            chunk_size = n // world_size
            remainder = n % world_size
            threshold = n - remainder
            for i, param in enumerate(group["params"]):
                if i >= threshold:
                    src_rank = world_size - 1
                else:
                    src_rank = i // chunk_size
                handle = dist.broadcast(param.data, src_rank, async_op=True)                
                handles.append(handle)

        for handle in handles:
            handle.wait()

        return loss
            

    def add_param_group(self, param_group: dict[str, Any]):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        params: list[nn.Parameter] = param_group["params"]
        n = len(params)
        chunk_size = n // world_size
        remainder = n % world_size
        if rank == world_size - 1:
            param_group["sharded_params"] = params[-(chunk_size + remainder):]
        else:
            param_group["sharded_params"] = params[rank * chunk_size : (rank + 1) * chunk_size]
        self.param_groups.append(param_group)