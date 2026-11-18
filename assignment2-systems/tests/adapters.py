from __future__ import annotations

import torch
import torch.distributed as dist

from cs336_systems.flash_attention import FlashAttention 
def get_flashattention_autograd_function_pytorch() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    The expectation is that this class will implement FlashAttention2
    using only standard PyTorch operations (no Triton!).

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyFlashAttnAutogradFunctionClass
    return FlashAttention

from cs336_systems.flash_attention import FlashAttentionTriton
def get_flashattention_autograd_function_triton() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_flashattention_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """
    # For example: return MyTritonFlashAttentionAutogradFunctionClass
    return FlashAttentionTriton

from cs336_systems.ddp import NaiveDDP
from cs336_systems.ddp import OverlapDDP
def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDP(module)
    
    # return NaiveDDP(module)
    return OverlapDDP(module)

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    
    # Naive DDP
    # for param in ddp_model.parameters():
    #     if param.grad is None:
    #         continue
    #     dist.all_reduce(param.grad, dist.ReduceOp.AVG, async_op=False)
    
    # grads = [param.grad for param in ddp_model.parameters() if param.grad is not None]
    # flattend_grads = _flatten_dense_tensors(grads)
    # dist.all_reduce(flattend_grads, dist.ReduceOp.AVG, async_op=False)
    # unflattend_grads = _unflatten_dense_tensors(flattend_grads, grads)
    # for old, new in zip(grads, unflattend_grads):
    #     old.copy_(new)

    # Overlap DDP
    ddp_model.finish_gradient_synchronization()


from cs336_systems.ddp import FSDP
from cs336_systems.ddp import gather_full_params

def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    return FSDP(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    fsdp_model.finish_gradient_synchronization()


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return gather_full_params(fsdp_model)

from cs336_systems.ddp import ShardedOptimizer
def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
