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

# FSDP claude的错误实现

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist


# class _ShardedParam:
#     """一个被分片的参数：本地 FP32 分片 + 还原完整形状所需的信息。"""

#     def __init__(self, shard: nn.Parameter, full_shape: torch.Size):
#         self.shard = shard                  # 沿 dim0 切分后的本地分片（FP32 主权重）
#         self.full_shape = full_shape        # 原始完整形状
#         self.shard_dim0 = shard.shape[0]    # 每个 rank 持有的行数（含 padding）


# class _Layer:
#     """一个被 FSDP 管理的 Linear / Embedding。"""

#     def __init__(self, idx: int, module: nn.Module):
#         self.idx = idx
#         self.module = module
#         self.params: dict[str, _ShardedParam] = {}
#         self.fwd_names: tuple = ()   # forward 需要 gather 的参数
#         self.bwd_names: tuple = ()   # backward 需要 gather 的参数


# class _FSDPLinearFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, weight_shard, bias_shard, fsdp, layer):
#         names = ("weight",) if bias_shard is None else ("weight", "bias")
#         full = fsdp._collect(layer, names)           # 等待/发起 all-gather
#         out = F.linear(x, full["weight"], full.get("bias"))
#         ctx.save_for_backward(x)                     # 只保存 x，不保存完整权重
#         ctx.fsdp, ctx.layer, ctx.has_bias = fsdp, layer, bias_shard is not None
#         del full                                     # 完整权重随函数返回被释放
#         fsdp._prefetch(layer.idx + 2, forward=True)  # 本层算完 -> 开始 gather 第 i+2 层
#         return out

#     @staticmethod
#     def backward(ctx, grad_out):
#         fsdp, layer = ctx.fsdp, ctx.layer
#         (x,) = ctx.saved_tensors
#         grad_out = grad_out.to(x.dtype)
#         fsdp._prefetch(layer.idx - 1, forward=False)       # 先发起上一层，与本层计算重叠

#         grad_x = None
#         if ctx.needs_input_grad[0]:
#             w = fsdp._collect(layer, ("weight",))["weight"]  # backward 重新 all-gather
#             grad_x = grad_out @ w
#             del w                                            # 用完立刻释放

#         go2d = grad_out.reshape(-1, grad_out.shape[-1])
#         x2d = x.reshape(-1, x.shape[-1])
#         fsdp._reduce_scatter(layer, "weight", go2d.t() @ x2d)
#         if ctx.has_bias:
#             fsdp._reduce_scatter(layer, "bias", go2d.sum(0))

#         fsdp._prefetch(layer.idx - 2, forward=False)
#         return grad_x, None, None, None, None


# class _FSDPEmbeddingFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, ids, weight_shard, fsdp, layer):
#         w = fsdp._collect(layer, ("weight",))["weight"]
#         out = F.embedding(ids, w, padding_idx=layer.module.padding_idx)
#         ctx.save_for_backward(ids)
#         ctx.fsdp, ctx.layer = fsdp, layer
#         del w
#         fsdp._prefetch(layer.idx + 2, forward=True)
#         return out

#     @staticmethod
#     def backward(ctx, grad_out):
#         fsdp, layer = ctx.fsdp, ctx.layer
#         (ids,) = ctx.saved_tensors
#         fsdp._prefetch(layer.idx - 1, forward=False)

#         info = layer.params["weight"]
#         num, dim = info.full_shape
#         flat_ids = ids.reshape(-1)
#         g = grad_out.reshape(-1, dim)
#         pad_idx = layer.module.padding_idx
#         if pad_idx is not None:
#             keep = flat_ids != pad_idx
#             flat_ids, g = flat_ids[keep], g[keep]
#         # embedding 的反向不需要完整权重，只需要 ids；用 fp32 累加避免重复 id 的精度损失
#         grad_w = torch.zeros(num, dim, dtype=torch.float32, device=grad_out.device)
#         grad_w.index_add_(0, flat_ids, g.float())
#         fsdp._reduce_scatter(layer, "weight", grad_w.to(grad_out.dtype))

#         fsdp._prefetch(layer.idx - 2, forward=False)
#         return None, None, None, None


# class FSDP(nn.Module):
#     def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
#         super().__init__()
#         self.module = module
#         self.compute_dtype = compute_dtype
#         self.rank = dist.get_rank()
#         self.world_size = dist.get_world_size()

#         self.layers: list[_Layer] = []
#         # layer_idx -> {param_name: (handle, gathered_buf, 通信用的输入分片)}
#         self._pending_gather: dict[int, dict] = {}
#         # [(handle, 分片梯度输出 buf, 对应的分片参数)]
#         self._pending_rs: list = []
#         # 未分片（replicated）参数的 all-reduce handle
#         self._pending_ar: list = []

#         self._sync_module_states()
#         self._shard_and_wrap()
#         self._register_replicated_grad_hooks()

#     # ------------------------------------------------------------------ 初始化
#     @torch.no_grad()
#     def _sync_module_states(self):
#         """保证各 rank 初始参数一致（以 rank0 为准）。"""
#         for t in list(self.module.parameters()) + list(self.module.buffers()):
#             dist.broadcast(t.data, src=0)

#     def _shard_param(self, p: nn.Parameter) -> _ShardedParam:
#         """沿 dim0 分片；dim0 不能整除时在末尾补 0 行。"""
#         ws, rank = self.world_size, self.rank
#         dim0 = p.shape[0]
#         k = -(-dim0 // ws)                       # ceil
#         data = p.data
#         if k * ws != dim0:
#             data = F.pad(data, [0, 0] * (data.ndim - 1) + [0, k * ws - dim0])
#         shard = data[rank * k:(rank + 1) * k].clone()   # clone：释放完整权重的显存
#         return _ShardedParam(nn.Parameter(shard, requires_grad=p.requires_grad), p.shape)

#     def _shard_and_wrap(self):
#         sharded: dict[int, _ShardedParam] = {}   # id(原参数) -> info，支持 tied weights
#         for m in self.module.modules():
#             if not isinstance(m, (nn.Linear, nn.Embedding)):
#                 continue
#             if isinstance(m, nn.Embedding):
#                 assert m.max_norm is None and not m.scale_grad_by_freq and not m.sparse, \
#                     "FSDP: Embedding 暂不支持 max_norm / scale_grad_by_freq / sparse"
#             layer = _Layer(len(self.layers), m)
#             for name, p in list(m.named_parameters(recurse=False)):
#                 if id(p) not in sharded:
#                     sharded[id(p)] = self._shard_param(p)
#                 info = sharded[id(p)]
#                 setattr(m, name, info.shard)     # 模块里换成分片（真正的 nn.Parameter）
#                 layer.params[name] = info
#             layer.fwd_names = tuple(layer.params.keys())
#             # Linear backward 只需要 weight；Embedding backward 不需要完整权重
#             layer.bwd_names = ("weight",) if isinstance(m, nn.Linear) else ()
#             self._wrap_forward(layer)
#             self.layers.append(layer)

#     def _wrap_forward(self, layer: _Layer):
#         m = layer.module
#         if isinstance(m, nn.Linear):
#             def fwd(x):
#                 if (self.compute_dtype is not None and x.is_floating_point()
#                         and x.dtype != self.compute_dtype):
#                     x = x.to(self.compute_dtype)
#                 return _FSDPLinearFn.apply(x, m.weight, m.bias, self, layer)
#         else:
#             def fwd(ids):
#                 return _FSDPEmbeddingFn.apply(ids, m.weight, self, layer)
#         m.forward = fwd

#     def _register_replicated_grad_hooks(self):
#         """没有被分片的参数（norm 等）：梯度做异步 all-reduce(AVG)。"""
#         sharded_ids = {id(info.shard) for l in self.layers for info in l.params.values()}
#         for p in self.module.parameters():
#             if id(p) in sharded_ids or not p.requires_grad:
#                 continue
#             p.register_post_accumulate_grad_hook(self._all_reduce_hook)

#     def _all_reduce_hook(self, p: torch.Tensor):
#         h = dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
#         self._pending_ar.append(h)

#     # ------------------------------------------------------------------ 通信
#     def _launch_gather(self, layer: _Layer, names):
#         """异步 all-gather：先 cast 成 compute_dtype，再通信。"""
#         pend = self._pending_gather.setdefault(layer.idx, {})
#         for n in names:
#             if n in pend:
#                 continue
#             info = layer.params[n]
#             src = info.shard.detach()
#             if self.compute_dtype is not None and src.dtype != self.compute_dtype:
#                 src = src.to(self.compute_dtype)
#             src = src.contiguous()
#             out = torch.empty((self.world_size * info.shard_dim0, *info.full_shape[1:]),
#                               dtype=src.dtype, device=src.device)
#             handle = dist.all_gather_into_tensor(out, src, async_op=True)
#             pend[n] = (handle, out, src)
#         if not pend:
#             del self._pending_gather[layer.idx]

#     def _collect(self, layer: _Layer, names) -> dict:
#         """拿到完整权重：没预取就现场发起，然后等待完成。返回去掉 padding 的视图。"""
#         self._launch_gather(layer, names)
#         pend = self._pending_gather.get(layer.idx, {})
#         full = {}
#         for n in names:
#             handle, out, _ = pend.pop(n)
#             handle.wait()
#             full[n] = out[:layer.params[n].full_shape[0]]
#         if layer.idx in self._pending_gather and not self._pending_gather[layer.idx]:
#             del self._pending_gather[layer.idx]
#         return full

#     def _prefetch(self, idx: int, forward: bool):
#         if 0 <= idx < len(self.layers):
#             layer = self.layers[idx]
#             self._launch_gather(layer, layer.fwd_names if forward else layer.bwd_names)

#     def _reduce_scatter(self, layer: _Layer, name: str, grad_full: torch.Tensor):
#         """异步 reduce-scatter(AVG)，结果在 finish_gradient_synchronization 里写入 shard.grad。"""
#         info = layer.params[name]
#         if not info.shard.requires_grad:
#             return
#         pad = self.world_size * info.shard_dim0 - grad_full.shape[0]
#         if pad:
#             grad_full = F.pad(grad_full, [0, 0] * (grad_full.ndim - 1) + [0, pad])
#         grad_full = grad_full.contiguous()
#         out = torch.empty((info.shard_dim0, *grad_full.shape[1:]),
#                           dtype=grad_full.dtype, device=grad_full.device)
#         handle = dist.reduce_scatter_tensor(out, grad_full, op=dist.ReduceOp.AVG, async_op=True)
#         # c10d 的 Work 对象/allocator 的 recordStream 会保证 grad_full 在通信完成前不被复用
#         self._pending_rs.append((handle, out, info.shard))

#     def _drain_gathers(self):
#         for pend in self._pending_gather.values():
#             for handle, _, _ in pend.values():
#                 handle.wait()
#         self._pending_gather.clear()

#     # ------------------------------------------------------------------ 对外接口
#     def forward(self, *inputs, **kwargs):
#         self._drain_gathers()                 # 清理上一轮遗留的预取
#         for layer in self.layers[:2]:         # 先 gather 第 0、1 层；之后每层算完再 gather 第 i+2 层
#             self._launch_gather(layer, layer.fwd_names)
#         return self.module(*inputs, **kwargs)

#     @torch.no_grad()
#     def finish_gradient_synchronization(self):
#         self._drain_gathers()
#         for handle, out, shard in self._pending_rs:
#             handle.wait()
#             g = out if out.dtype == shard.dtype else out.to(shard.dtype)  # 梯度转回 FP32
#             if shard.grad is None:
#                 shard.grad = g
#             else:
#                 shard.grad += g               # 支持梯度累积 / tied weights
#         self._pending_rs.clear()
#         for h in self._pending_ar:
#             h.wait()
#         self._pending_ar.clear()


# @torch.no_grad()
# def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
#     """
#     All-gather sharded parameters from the FSDP model to reconstruct full
#     parameter tensors. Replicated parameters are returned as-is.
#     """
#     # 不是 FSDP 包装的模型：所有参数本来就是完整的
#     if not isinstance(fsdp_model, FSDP):
#         return {n: p.detach().clone() for n, p in fsdp_model.named_parameters()}

#     # 先等待所有未完成的通信，避免和预取 / reduce-scatter 交错
#     fsdp_model._drain_gathers()

#     # id(本地分片参数) -> 分片信息，用于识别哪些参数是被分片的
#     shard_info = {
#         id(info.shard): info
#         for layer in fsdp_model.layers
#         for info in layer.params.values()
#     }

#     full_cache: dict[int, torch.Tensor] = {}   # tied weights 只 gather 一次
#     result: dict[str, torch.Tensor] = {}

#     # remove_duplicate=False：共享权重的每个名字都出现在结果里
#     # 迭代顺序在所有 rank 上一致，保证 collective 调用顺序相同
#     for name, p in fsdp_model.module.named_parameters(remove_duplicate=False):
#         info = shard_info.get(id(p))
#         if info is None:
#             # 未分片（replicated）参数，原样返回
#             result[name] = p.detach().clone()
#             continue

#         key = id(p)
#         if key not in full_cache:
#             src = p.detach().contiguous()   # 本地 FP32 主权重，不经过 compute_dtype，精度无损
#             out = torch.empty((fsdp_model.world_size * info.shard_dim0, *info.full_shape[1:]),
#                               dtype=src.dtype, device=src.device)
#             dist.all_gather_into_tensor(out, src)             # 同步 all-gather
#             full_cache[key] = out[:info.full_shape[0]].clone()  # 去掉 padding，并释放大 buffer
#         result[name] = full_cache[key]

#     return result

