import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

from functools import partial

import torch
import torch.distributed as dist
from custom_linear_A import CustomLinear as CLA
from custom_linear_B import CustomLinear as CLB
from data_utils import DataLoader, FullCtx
from datasets import load_dataset
from llama_hess import LlamaForCausalLM
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch
from torch.distributed.fsdp.wrap import (enable_wrap,
                                         transformer_auto_wrap_policy, wrap)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

import argparse

import tqdm
from accelerate import init_empty_weights
from lightning.pytorch.utilities.seed import isolate_rng

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--n_seqs', default=65536, type=int)
parser.add_argument('--ctx_size', default=2048, type=int)
parser.add_argument('--power_iters', default=1, type=int)
parser.add_argument('--start_layer', default=0, type=int)
parser.add_argument('--end_layer', default=100000, type=int)
parser.add_argument('--hessian_sketch', default='B', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--orig_model', type=str)
parser.add_argument('--cpu_offload', action='store_true')
parser.add_argument('--fp64_accum', action='store_true')
args = parser.parse_args()


def setup(rank, world_size):
    from datetime import timedelta
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(days=1))


def cleanup():
    dist.destroy_process_group()


local_rank = int(os.environ["LOCAL_RANK"])
local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

setup(local_rank, local_world_size)

cutoff = args.n_seqs // (local_world_size * args.batch_size)
if local_rank == 0:
    print(f'USING {cutoff} SEQUENCES PER GPU')

model = LlamaForCausalLM.from_pretrained(args.orig_model,
                                         torch_dtype='auto',
                                         device_map='cpu')

if args.hessian_sketch == 'A':
    custom_linear_layer = CLA
    if args.power_iters < 2:
        if local_rank == 0:
            print(
                'ERROR: Must use more than half a round of power iteration for A'
            )
        raise Exception
elif args.hessian_sketch == 'B':
    custom_linear_layer = CLB
else:
    raise Exception

device_ct = 0
with torch.autograd.set_grad_enabled(False):
    for i in range(len(model.model.layers)):
        l = model.model.layers[i]
        collect_hess = (i >= args.start_layer and i < args.end_layer)
        args.fp64_accum = False

        name = f'{i}_q'
        new_q = custom_linear_layer(device_ct % local_world_size,
                                    args.cpu_offload,
                                    os.path.join(args.save_path, name),
                                    collect_hess,
                                    args.fp64_accum,
                                    l.self_attn.q_proj.in_features,
                                    l.self_attn.q_proj.out_features,
                                    dtype=l.self_attn.q_proj.weight.dtype)
        new_q.weight = l.self_attn.q_proj.weight
        del l.self_attn.q_proj
        l.self_attn.q_proj = new_q
        device_ct += 1

        name = f'{i}_k'
        new_k = custom_linear_layer(device_ct % local_world_size,
                                    args.cpu_offload,
                                    os.path.join(args.save_path, name),
                                    collect_hess,
                                    args.fp64_accum,
                                    l.self_attn.k_proj.in_features,
                                    l.self_attn.k_proj.out_features,
                                    dtype=l.self_attn.k_proj.weight.dtype)
        new_k.weight = l.self_attn.k_proj.weight
        del l.self_attn.k_proj
        l.self_attn.k_proj = new_k
        device_ct += 1

        name = f'{i}_v'
        new_v = custom_linear_layer(device_ct % local_world_size,
                                    args.cpu_offload,
                                    os.path.join(args.save_path, name),
                                    collect_hess,
                                    args.fp64_accum,
                                    l.self_attn.v_proj.in_features,
                                    l.self_attn.v_proj.out_features,
                                    dtype=l.self_attn.v_proj.weight.dtype)
        new_v.weight = l.self_attn.v_proj.weight
        del l.self_attn.v_proj
        l.self_attn.v_proj = new_v
        device_ct += 1

        name = f'{i}_o'
        new_o = custom_linear_layer(device_ct % local_world_size,
                                    args.cpu_offload,
                                    os.path.join(args.save_path, name),
                                    collect_hess,
                                    args.fp64_accum,
                                    l.self_attn.o_proj.in_features,
                                    l.self_attn.o_proj.out_features,
                                    dtype=l.self_attn.o_proj.weight.dtype)
        new_o.weight = l.self_attn.o_proj.weight
        del l.self_attn.o_proj
        l.self_attn.o_proj = new_o
        device_ct += 1

        name = f'{i}_up'
        new_up = custom_linear_layer(device_ct % local_world_size,
                                     args.cpu_offload,
                                     os.path.join(args.save_path, name),
                                     collect_hess,
                                     args.fp64_accum,
                                     l.mlp.up_proj.in_features,
                                     l.mlp.up_proj.out_features,
                                     dtype=l.mlp.up_proj.weight.dtype)
        new_up.weight = l.mlp.up_proj.weight
        del l.mlp.up_proj
        l.mlp.up_proj = new_up
        device_ct += 1

        name = f'{i}_gate'
        new_gate = custom_linear_layer(device_ct % local_world_size,
                                       args.cpu_offload,
                                       os.path.join(args.save_path, name),
                                       collect_hess,
                                       args.fp64_accum,
                                       l.mlp.gate_proj.in_features,
                                       l.mlp.gate_proj.out_features,
                                       dtype=l.mlp.gate_proj.weight.dtype)
        new_gate.weight = l.mlp.gate_proj.weight
        del l.mlp.gate_proj
        l.mlp.gate_proj = new_gate
        device_ct += 1

        name = f'{i}_down'
        new_down = custom_linear_layer(device_ct % local_world_size,
                                       args.cpu_offload,
                                       os.path.join(args.save_path, name),
                                       collect_hess,
                                       args.fp64_accum,
                                       l.mlp.down_proj.in_features,
                                       l.mlp.down_proj.out_features,
                                       dtype=l.mlp.down_proj.weight.dtype)
        new_down.weight = l.mlp.down_proj.weight
        del l.mlp.down_proj
        l.mlp.down_proj = new_down
        device_ct += 1

auto_wrap_policy = partial(transformer_auto_wrap_policy,
                           transformer_layer_cls={
                               type(model.model.layers[0]),
                           })

model = FSDP(model,
             device_id=local_rank,
             auto_wrap_policy=auto_wrap_policy,
             cpu_offload=CPUOffload(offload_params=True),
             use_orig_params=False)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=non_reentrant_wrapper,
    check_fn=(lambda module: not type(module) == custom_linear_layer),
)

torch.cuda.empty_cache()

tok = AutoTokenizer.from_pretrained(args.orig_model)

batch = torch.zeros(args.batch_size,
                    args.ctx_size,
                    dtype=torch.int64,
                    device=local_rank)

for pit in range(args.power_iters):

    if local_rank == 0:
        print(f'POWER ITERATION {pit}')
        dataset = load_dataset('togethercomputer/RedPajama-Data-1T-Sample',
                               split='train').shuffle(args.seed)
        dl = iter(
            torch.utils.data.DataLoader(
                FullCtx(iter(dataset), tok, args.ctx_size),
                batch_size=args.batch_size * local_world_size,
                num_workers=1))

    range_counter = range(cutoff)
    if local_rank == 0:
        range_counter = tqdm.tqdm(range_counter)
    for i in range_counter:
        blist = list(
            torch.split(next(dl).to(local_rank), args.batch_size,
                        dim=0)) if local_rank == 0 else None

        torch.distributed.scatter(batch, blist, src=0)
        logits = model(batch,
                       mode=(pit, i == 0, i == (cutoff - 1)),
                       use_cache=False)['logits']
        logits = logits.view(-1, logits.shape[-1]).float()

        with torch.no_grad():
            with isolate_rng():
                torch.manual_seed(i)
                fake_target = torch.distributions.categorical.Categorical(
                    logits=logits).sample()

        torch.nn.functional.cross_entropy(logits, fake_target).backward()

        if i == cutoff - 1:
            for l in model.modules():
                if hasattr(l, 'hin'):
                    ct = max(l.ct, 1)
                    torch.save(
                        l.hin / ct,
                        os.path.join(args.save_path, f'{l.fname}_hin.pt'))
                    torch.save(
                        l.hout / ct,
                        os.path.join(args.save_path, f'{l.fname}_hout.pt'))

            print(f'RANK {local_rank} SAVED CURRENT HESSIANS')
