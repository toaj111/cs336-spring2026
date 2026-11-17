import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

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