from __future__ import annotations
import torch
import torch.distributed as dist
import torch.nn as nn
from cs336_basics.model import Embedding, Linear
from einops import einsum
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



"""Fully Sharded Data Parallel for cs336_basics Linear and Embedding layers."""
# TODO: 实现异步all gather
def _pad_rows(weight: torch.Tensor, world_size: int) -> tuple[torch.Tensor, int]:
    """Pad row count to a multiple of ``world_size`` so every rank gets an equal shard."""
    original_rows = weight.shape[0]
    remainder = original_rows % world_size
    if remainder == 0:
        return weight, original_rows
    padding = weight.new_zeros((world_size - remainder, *weight.shape[1:]))
    return torch.cat((weight, padding), dim=0), original_rows


def _gather_full_weight(weight_shard: torch.Tensor, world_size: int) -> torch.Tensor:
    shards = [torch.empty_like(weight_shard) for _ in range(world_size)]
    dist.all_gather(shards, weight_shard)
    return torch.cat(shards, dim=0)


def _reduce_to_shard(grad_full: torch.Tensor, world_size: int) -> torch.Tensor:
    grad_full = grad_full.contiguous()
    dist.all_reduce(grad_full, op=dist.ReduceOp.SUM)
    grad_full.div_(world_size)
    return grad_full.chunk(world_size, dim=0)[dist.get_rank()].contiguous()


def _gathered_weight_with_hook(
    weight_shard: nn.Parameter,
    world_size: int,
    compute_dtype: torch.dtype | None,
    full_rows: int,
) -> torch.Tensor:
    with torch.no_grad():
        padded_weight = _gather_full_weight(weight_shard, world_size)
    padded_rows = padded_weight.shape[0]
    full_weight = padded_weight[:full_rows].detach().requires_grad_(True)

    def _shard_grad_hook(grad_full: torch.Tensor) -> None:
        if grad_full.shape[0] < padded_rows:
            missing = padded_rows - grad_full.shape[0]
            grad_full = torch.cat(
                (grad_full, grad_full.new_zeros((missing, *grad_full.shape[1:]))),
                dim=0,
            )
        grad_shard = _reduce_to_shard(grad_full, world_size)
        if compute_dtype is not None:
            grad_shard = grad_shard.to(torch.float32)
        if weight_shard.grad is None:
            weight_shard.grad = grad_shard
        else:
            weight_shard.grad.add_(grad_shard)

    full_weight.register_hook(_shard_grad_hook)
    if compute_dtype is not None:
        return full_weight.to(compute_dtype)
    return full_weight


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        self._sharded_modules: dict[str, nn.Module] = {}
        self._wrap_sharded_layers()

    def _wrap_sharded_layers(self) -> None:
        for name, module in self.module.named_modules():
            if isinstance(module, Linear):
                self._shard_module(name, module, linear=True)
            elif isinstance(module, Embedding):
                self._shard_module(name, module, linear=False)

    def _shard_module(self, name: str, module: nn.Module, *, linear: bool) -> None:
        full_weight = module.weight
        if not isinstance(full_weight, nn.Parameter):
            raise TypeError(f"{name} weight is not a parameter")
        padded_weight, full_rows = _pad_rows(full_weight.detach(), self._world_size)
        shard_rows = padded_weight.shape[0] // self._world_size
        start = self._rank * shard_rows
        module.weight = nn.Parameter(padded_weight[start : start + shard_rows].clone())
        setattr(module, "_fsdp_full_rows", full_rows)
        self._sharded_modules[name] = module

        def _weight_parameter(mod: nn.Module) -> nn.Parameter:
            weight = mod.weight
            if not isinstance(weight, nn.Parameter):
                raise TypeError("sharded weight is not a parameter")
            return weight

        if linear:

            def linear_forward(x: torch.Tensor, mod: nn.Module = module) -> torch.Tensor:
                weight = _gathered_weight_with_hook(
                    _weight_parameter(mod), self._world_size, self.compute_dtype, full_rows
                )
                return einsum(x, weight, "... d_in, d_out d_in -> ... d_out")

            module.forward = linear_forward  # type: ignore[method-assign]
        else:

            def embedding_forward(token_ids: torch.Tensor, mod: nn.Module = module) -> torch.Tensor:
                weight = _gathered_weight_with_hook(
                    _weight_parameter(mod), self._world_size, self.compute_dtype, full_rows
                )
                return weight[token_ids, :]

            module.forward = embedding_forward  # type: ignore[method-assign]

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for name, param in self.module.named_parameters():
            module_name = name.rsplit(".", 1)[0]
            if module_name in self._sharded_modules:
                continue
            if param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(self._world_size)


def gather_full_params(fsdp_model: FSDP) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, param in fsdp_model.module.named_parameters():
        module_name = name.rsplit(".", 1)[0]
        if module_name not in fsdp_model._sharded_modules:
            result[name] = param.data.clone()
            continue
        shard_shape = param.shape
        shards = [torch.empty(shard_shape, device=param.device, dtype=param.dtype) for _ in range(fsdp_model._world_size)]
        dist.all_gather(shards, param.data)
        gathered = torch.cat(shards, dim=0)
        full_rows = getattr(fsdp_model._sharded_modules[module_name], "_fsdp_full_rows", gathered.shape[0])
        result[name] = gathered[:full_rows].contiguous()
    return result


# FSDP错误实现
# class AllGatherFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, shard_param: torch.Tensor, tensor_list):
#         ctx.shard_shape = shard_param.shape
#         full_param = torch.cat(tensor_list, dim=0)
#         return full_param

#     @staticmethod
#     def backward(ctx, grad_output: torch.Tensor):
#         world_size = dist.get_world_size()
#         grad_chunks = torch.chunk(grad_output, world_size, dim=0)
#         grad_shard = torch.empty(ctx.shard_shape, dtype=grad_output.dtype, device=grad_output.device)
#         dist.reduce_scatter(grad_shard, list(grad_chunks), dist.ReduceOp.AVG)
#         return grad_shard, None

# class FSDP(nn.Module):
#     def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
#         super().__init__()
#         self.module = module
#         self.compute_dtype = compute_dtype
#         self.sharded_param_names = []
#         self.rank = dist.get_rank()
#         self.world_size = dist.get_world_size()
#         self.layers = self._get_layers(module)
#         self._init_sharding()
#         self._reigister_hooks()
#         self.handles = {}
#         self.gathered_params = {}

#     # 按顺序所有layer（假设计算顺序和注册顺序一致）
#     def _get_layers(self, module: nn.Module) -> list[nn.Module]:
#         out = []
#         for _, child in module.named_children():
#             if isinstance(child, (nn.Sequential, nn.ModuleList, Iterable)):
#                 out += self._get_layers(child)
#             else:
#                 out.append(child)
#         return out

#     # 异步收集指定层的参数
#     def _aync_gather_layers(self, layer_idx: int):
#         if layer_idx >= len(self.layers):
#             return
#         layer = self.layers[layer_idx]
#         layer_handles = []
#         layer_full_params = {}

#         for name, param in layer.named_parameters():
#             if name.startswith("norm"):
#                 continue
#             if param.ndim >= 2 and param.shape[0] % self.world_size == 0:
#                 tensor_list = [torch.empty_like(param) for _ in range(self.world_size)]
#                 handle = dist.all_gather(tensor_list, param.contiguous(), async_op=True)
#                 layer_handles.append(handle)
#                 layer_full_params[name] = (tensor_list, param)

#             self.handles[layer_idx] = layer_handles
#             self.gathered_params[layer_idx] = layer_full_params

#     # 初始化分片
#     def _init_sharding(self):
#         for name, param in self.module.named_parameters():
#             if name.startswith("norm"):
#                 continue
#             if param.ndim >= 2 and param.shape[0] % self.world_size == 0:
#                 self.sharded_param_names.append(name)
#                 full_data = param.data
#                 chunks = torch.chunk(full_data, self.world_size, dim=0)
#                 local_data = chunks[self.rank].contiguous()
#                 *mod_path, param_name = name.split('.') #a.b.weight
#                 target_mod = self.module
#                 # 找到最终的目的module
#                 for p in mod_path:
#                     target_mod = getattr(target_mod, p)
#                 setattr(target_mod, param_name, nn.Parameter(local_data))

#     # 初始化hook
#     def _reigister_hooks(self):
#         for idx, layer in enumerate(self.layers):
#             def make_pre_hook(i):
#                 def pre_hook(mod, args):
#                     # 等待当前层异步通信完成
#                     if i in self.handles:
#                         for handle in self.handles[i]:
#                             handle.wait()

#                         for name, (tensor_list, shard_param) in self.gathered_params[i].items():
#                             *mod_path, param_name = name.split('.')
#                             target_mod = mod
#                             for p in mod_path:
#                                 target_mod = getattr(target_mod, p)

#                             full_param = AllGatherFunction.apply(shard_param, tensor_list)
#                             if self.compute_dtype is not None:
#                                 full_param = full_param.to(self.compute_dtype)
#                             setattr(target_mod, param_name, full_param)
#                 return pre_hook
            
#             def make_post_hook(i):
#                 def post_hook(mod, args, output):
#                     if i in self.gathered_params:
#                         for name, (_, shard_param) in self.gathered_params[i].items():
#                             *mod_path, param_name = name.split('.')
#                             target_mod = mod
#                             for p in mod_path:
#                                 target_mod = getattr(target_mod, p)
#                             setattr(target_mod, param_name, shard_param)
#                             del self.gathered_params[i]
#                             del self.handles[i]

#                     self._aync_gather_layers(i + 2)
#                 return post_hook

#             layer.register_forward_pre_hook(make_pre_hook(idx))
#             layer.register_forward_hook(make_post_hook(idx))



#     def forward(self, *inputs, **kwargs):
#         self.handles.clear()
#         self.gathered_params.clear()
#         if len(self.layers) > 0:
#             self._aync_gather_layers(0)
#         if len(self.layers) > 1:
#             self._aync_gather_layers(1)
#         return self.module(*inputs, **kwargs)

#     def finish_gradient_synchronization(self):
#         for _, layer_handles in self.handles.items():
#             for handle in layer_handles:
#                 handle.wait()
#             layer_handles.clear()



