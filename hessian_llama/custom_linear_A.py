import glob
import os

import torch
import torch.distributed as dist
import torch.nn as nn

torch._dynamo.config.cache_size_limit = 256

local_rank = int(os.environ.get("LOCAL_RANK", 0))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

from torch.distributed import ReduceOp


@torch.compile
def sym_to_flat(A):
    N = A.shape[-1]
    idxs = torch.tril_indices(N, N, device=A.device)
    return A[idxs.unbind()]


@torch.compile
def flat_to_sym(V, N):
    A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
    idxs = torch.tril_indices(N, N, device=V.device)
    A[idxs.unbind()] = V
    A[idxs[1, :], idxs[0, :]] = V
    return A


class LinearNoBias(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, input, weight, mode, parent_class):
        ctx.save_for_backward(input, weight)
        ctx.mode = mode
        ctx.parent_class = parent_class

        return input @ weight.T

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        it, reset, div = ctx.mode
        is_buffer = local_rank == ctx.parent_class.buffer_dev

        input, weight = ctx.saved_tensors
        ws = weight.shape
        grad_input = grad_output @ weight
        del weight

        if ctx.parent_class.collect_hess:
            grad_output = grad_output.reshape(-1, grad_output.shape[-1])
            input = input.reshape(-1, input.shape[-1])
            op_dtype = ctx.parent_class.op_dtype
            with torch.amp.autocast('cuda', enabled=False):
                grad_output = grad_output.float()
                input = input.float()
                bs = input.shape[0]
                if it == 0:
                    in_hess = sym_to_flat(
                        input.T @ input) / ctx.parent_class.scale
                    handle = torch.distributed.reduce(
                        in_hess,
                        ctx.parent_class.buffer_dev,
                        op=ReduceOp.AVG,
                        async_op=True)
                    if reset and is_buffer:
                        ctx.parent_class.hin.mul_(0)
                    del input, grad_output
                    handle.wait()
                    if is_buffer:
                        ctx.parent_class.hin.add_(
                            in_hess.to(
                                ctx.parent_class.hin.device).to(op_dtype))
                        ctx.parent_class.ct += bs / ctx.parent_class.scale
                        if div:
                            ctx.parent_class.hin.div_(ctx.parent_class.ct)
                            ctx.parent_class.ct = 0

                    del in_hess
                    torch.cuda.empty_cache()
                else:
                    if it % 2 == 0:
                        if reset and is_buffer:
                            ctx.parent_class.hin.mul_(0)
                        if not is_buffer:
                            out_hess = torch.zeros(
                                ctx.parent_class.out_features *
                                (ctx.parent_class.out_features + 1) // 2,
                                dtype=op_dtype,
                                device=local_rank)
                        else:
                            out_hess = ctx.parent_class.hout.to(local_rank)
                        torch.distributed.broadcast(
                            out_hess, ctx.parent_class.buffer_dev)
                        out_hess = flat_to_sym(out_hess, ws[0]).float()
                        in_hess = input.T @ (input * (
                            (grad_output @ out_hess) * grad_output).sum(
                                dim=-1, keepdims=True)) / out_hess.norm()**2
                        del input, grad_output, out_hess
                        in_hess = sym_to_flat(in_hess) / ctx.parent_class.scale
                        torch.distributed.reduce(in_hess,
                                                 ctx.parent_class.buffer_dev,
                                                 op=ReduceOp.AVG)
                        if is_buffer:
                            ctx.parent_class.hin.add_(
                                in_hess.to(
                                    ctx.parent_class.hin.device).to(op_dtype))
                            ctx.parent_class.ct += bs / ctx.parent_class.scale
                            if div:
                                ctx.parent_class.hin.div_(ctx.parent_class.ct)
                                ctx.parent_class.ct = 0

                        del in_hess
                    else:
                        if reset and is_buffer:
                            ctx.parent_class.hout.mul_(0)
                        if not is_buffer:
                            in_hess = torch.zeros(
                                ctx.parent_class.in_features *
                                (ctx.parent_class.in_features + 1) // 2,
                                dtype=op_dtype,
                                device=local_rank)
                        else:
                            in_hess = ctx.parent_class.hin.to(local_rank)
                        torch.distributed.broadcast(
                            in_hess, ctx.parent_class.buffer_dev)
                        in_hess = flat_to_sym(in_hess, ws[1]).float()
                        out_hess = grad_output.T @ (grad_output * (
                            (input @ in_hess) * input).sum(
                                dim=-1, keepdims=True)) / in_hess.norm()**2
                        del input, grad_output, in_hess
                        out_hess = sym_to_flat(
                            out_hess) / ctx.parent_class.scale
                        torch.distributed.reduce(out_hess,
                                                 ctx.parent_class.buffer_dev,
                                                 op=ReduceOp.AVG)
                        if is_buffer:
                            ctx.parent_class.hout.add_(
                                out_hess.to(
                                    ctx.parent_class.hout.device).to(op_dtype))
                            ctx.parent_class.ct += bs / ctx.parent_class.scale
                            if div:
                                ctx.parent_class.hout.div_(ctx.parent_class.ct)
                                ctx.parent_class.ct = 0

                        del out_hess

        torch.cuda.empty_cache()
        return grad_input.to(local_rank), None, None, None


class CustomLinear(nn.Linear):

    def __init__(self,
                 buffer_dev,
                 cpu_offload,
                 load_fname,
                 collect_hess=True,
                 use_fp64=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.fname = load_fname
        self.collect_hess = collect_hess
        self.op_dtype = torch.float32 if not use_fp64 else torch.float64
        self.scale = 8192
        if collect_hess and local_rank == buffer_dev:
            device = 'cpu' if cpu_offload else buffer_dev
            last_it = sorted(glob.glob(f'{load_fname}_hin*.pt'))
            if len(last_it) > 0 and os.path.exists(last_it[-1]):
                self.hin = torch.load(last_it[-1],
                                      map_location=torch.device(device)).to(
                                          self.op_dtype)
                print(f'loaded from {last_it[-1]}')
            else:
                self.hin = torch.zeros(self.in_features *
                                       (self.in_features + 1) // 2,
                                       dtype=self.op_dtype,
                                       device=device)
            last_it = sorted(glob.glob(f'{load_fname}_hout*.pt'))
            if len(last_it) > 0 and os.path.exists(last_it[-1]):
                self.hout = torch.load(last_it[-1],
                                       map_location=torch.device(device)).to(
                                           self.op_dtype)
                print(f'loaded from {last_it[-1]}')
            else:
                self.hout = torch.zeros(self.out_features *
                                        (self.out_features + 1) // 2,
                                        dtype=self.op_dtype,
                                        device=device)
            if cpu_offload:
                self.hin.pin_memory()
                self.hout.pin_memory()

        self.buffer_dev = buffer_dev
        self.ct = 0

    def forward(self, input, mode):
        return LinearNoBias.apply(input, self.weight, mode, self)

    def reset_parameters(self):
        return
