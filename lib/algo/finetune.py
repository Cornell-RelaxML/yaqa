"""
Utilities for fine tuning
"""
import copy
import math
import os
from contextlib import contextmanager
from functools import partial
from operator import attrgetter

import glog
import torch
from adam import Adam as EigAdam
from torch import multiprocessing as mp
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

from lib import codebook, utils
from lib.linear import QuantizedLinear

from . import ldlq


@contextmanager
def use_tf32():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)


def finetune_decoder_layer(layer, attr_mapping, name, device, train_dl,
                           valid_dl, orig_dtype, position_embeddings, args):
    with use_tf32():
        layer = layer.to(device)

        source = next(iter(train_dl))[0]
        position_ids = torch.arange(source.shape[1],
                                    device=device).unsqueeze(0)
        # manifest tensor parallel attributes in layer
        output = layer(source.to(device),
                       position_ids=position_ids,
                       position_embeddings=position_embeddings)[0]

        best_sd = {k: v.cpu() for k, v in layer.state_dict().items()}
        utils.clean()

        optim = torch.optim.Adam(layer.parameters(), lr=args.ft_lr, eps=1e-12)

        best_loss = utils.calculate_mse_loss(layer, valid_dl, device,
                                             position_embeddings)
        glog.info(f'layer {name} initial loss {best_loss}')
        scaler = torch.cuda.amp.GradScaler(
            enabled=(orig_dtype == torch.float16))
        worse_ct = 0

        for epoch in range(args.ft_epochs):
            for bidx, (source, targets) in enumerate(train_dl):
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda',
                                    dtype=orig_dtype,
                                    enabled=True):
                    output = layer(source.to(device),
                                   position_ids=position_ids,
                                   position_embeddings=position_embeddings)[0]
                    loss = nn.MSELoss()(output,
                                        targets) / targets.square().mean()

                scaler.scale(loss).backward()

                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                        train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                test_loss = utils.calculate_mse_loss(layer, valid_dl, device,
                                                     position_embeddings)
                if test_loss < best_loss:
                    glog.info(
                        f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                    )
                    best_loss = test_loss
                    best_sd = {
                        k: v.cpu()
                        for k, v in layer.state_dict().items()
                    }
                    utils.clean()
                    worse_ct = 0
                else:
                    glog.info(
                        f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                    )
                    worse_ct += 1
                    if worse_ct >= args.ft_early_stop:
                        break

    del optim, train_dl, valid_dl

    layer = layer.cpu()
    layer.load_state_dict(best_sd)
    utils.clean()


def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args,
                                    device, pre_orig_emb, orig_emb,
                                    position_embeddings):
    if args.ft_grad_ckpt:
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            mixed_layer,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda module: isinstance(module, nn.Linear),
        )

    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)
    torch.set_grad_enabled(False)

    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    orig_dtype = None
    for p in mixed_layer.parameters():
        orig_dtype = p.dtype
        break
    mixed_layer = mixed_layer.float()

    train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    has_kernel = utils.has_kernel(args.decode_mode, args.L, args.K, args.V,
                                  args.tlut_bits, args.td_x, args.td_y)

    for quant_i, attr in enumerate(quant_order):
        (linear_attr, name, in_hess_name, out_hess_name, rcp, skip_ft,
         one_sided_override) = attr
        save_path = f'{args.save_path}/{idx}_{name}.pt'

        utils.clean()
        cb = cb.to(device).to(orig_dtype)
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        W = orig_linear.weight.to(dtype_)
        (m, n) = W.shape
        SU = (torch.randn(n, device=device).sign() +
              1e-5).sign().to(dtype_).to(device)
        SV = (torch.randn(m, device=device).sign() +
              1e-5).sign().to(dtype_).to(device)

        in_hess_path = f'{args.hess_path}/{idx}_{in_hess_name}_hin.pt'
        out_hess_path = f'{args.hess_path}/{idx}_{out_hess_name}_hout.pt'

        H_data = torch.load(in_hess_path,
                            map_location=torch.device('cpu')).to(device)
        Hin = utils.flat_to_sym(H_data, n).to(torch.float64)
        Hin /= torch.diag(Hin).mean()
        H_data = torch.load(out_hess_path,
                            map_location=torch.device('cpu')).to(device)
        Hout = utils.flat_to_sym(H_data, m).to(torch.float64)
        Hout /= torch.diag(Hout).mean()
        del H_data

        Hin = utils.matmul_hadUt(utils.matmul_hadUt(Hin * SU).T * SU).T
        Lin = None
        fsr = 0
        while Lin is None:
            Hin[torch.arange(n), torch.arange(n)] += args.sigma_reg
            fsr += args.sigma_reg
            Lin = utils.block_LDL(Hin, args.td_y)
        print('Final Sigma Reg Hin', fsr)
        Lin = Lin[0].float()
        Lin[torch.arange(n), torch.arange(n)] = 0

        Hout = utils.matmul_hadUt(utils.matmul_hadUt(Hout * SV).T * SV).T
        Lout = None
        fsr = 0
        while Lout is None:
            Hout[torch.arange(m), torch.arange(m)] += args.sigma_reg
            fsr += args.sigma_reg
            Lout = utils.block_LDL(Hout, args.td_x)
        print('Final Sigma Reg Hout', fsr)
        Lout = Lout[0].float()
        Lout[torch.arange(m), torch.arange(m)] = 0

        Hin = Hin.float()
        Hout = Hout.float()

        Wr = utils.matmul_hadUt(utils.matmul_hadUt(W.T.to(device) * SV).T * SU)

        Wscale = Wr.square().mean().sqrt() / (
            cb.lut.to(torch.float64).square().mean().sqrt().float() *
            args.scale_override)
        Wr /= Wscale

        hatWr, Qidxs = ldlq.LDLQ_2hess(Wr,
                                       Lin,
                                       Lout,
                                       args.td_x,
                                       args.td_y,
                                       args.V,
                                       cb,
                                       for_kernel=has_kernel)
        utils.clean()

        Qidxs = Qidxs.cpu()
        packed = cb.pack_trellis(
            Qidxs.reshape(m // args.td_x, args.td_x, n // args.td_y,
                          args.td_y // args.V).transpose(1, 2).reshape(
                              -1, args.td_x * args.td_y // args.V))

        if has_kernel:
            packed = packed.view(torch.uint8).view(-1, 2).flip(
                (-1, )).reshape(m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8,
                                args.K).permute(0, 2, 4, 3, 1, 5).flip(
                                    (-1, )).contiguous().flatten().view(
                                        torch.int16).reshape(packed.shape)
        else:
            packed = packed.view(torch.int16)

        Wr *= Wscale
        hatWr *= Wscale

        hatW = (utils.matmul_hadU((utils.matmul_hadU(hatWr) * SU).T) * SV).T
        W = W.to(device)
        err_num = torch.trace((Wr - hatWr) @ Hin @ (Wr - hatWr).T @ Hout)
        err = err_num / torch.trace(Wr @ Hin @ Wr.T @ Hout)
        print(f'{idx}_{name} 2 sided proxy err {err.item()}', err_num)

        err_num_1 = torch.trace((Wr - hatWr) @ Hin @ (Wr - hatWr).T)
        denom_1 = torch.trace(Wr @ Hin @ Wr.T)
        err_1 = err_num_1 / denom_1
        print(f'{idx}_{name} 1 sided proxy err {err_1.item()}', err_num_1)

        # 0 = no tensor parallelism, 1 = row parallel, 2 = column parallel
        rcp_int = 0

        torch.save(
            {
                'trellis':
                packed.cpu(),
                'SU':
                SU.to(orig_dtype).cpu(),
                'SV':
                SV.to(orig_dtype).cpu(),
                'Wscale':
                Wscale,
                'proxy_err':
                err_num.item(),
                'tlut':
                cb.tlut.data.to(orig_dtype).cpu()
                if hasattr(cb, 'tlut') else None,
                'rcp':
                rcp_int,
                'tp_rank':
                args.tp_rank
            }, save_path)

        del Hin, Hout, Lin, Lout, Wr, hatWr, Qidxs
        utils.clean()

        q_linear = QuantizedLinear(
            n,
            m,
            args.td_x,
            args.td_y,
            args.L,
            args.K,
            args.V,
            args.tlut_bits,
            args.decode_mode,
            mode='train-recons' if args.ft_train_lut else 'train-fixW',
            dtype=orig_dtype,
            grad_ckpt=args.ft_grad_ckpt)
        q_linear.trellis.copy_(packed)
        q_linear.SU.copy_(SU)
        q_linear.SV.copy_(SV)
        q_linear.rcp.copy_(rcp_int)
        q_linear.tp_rank.copy_(args.tp_rank)
        q_linear = q_linear.to(device).float()

        del packed, SU, SV
        utils.clean()

        q_linear.SU = nn.Parameter(q_linear.SU, requires_grad=True)
        q_linear.SV = nn.Parameter(q_linear.SV * Wscale, requires_grad=True)

        if q_linear.tlut is not None:
            q_linear.tlut.copy_(cb.tlut.data)
            q_linear.tlut.requires_grad = args.ft_train_lut

        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            q_linear)

        if args.ft_epochs > 0 and not skip_ft:
            attr_map = []
            for attr in quant_order[quant_i + 1:]:
                (linear_attr, _, in_hess_name, out_hess_name, _, _, _) = attr
                attr_map.append(
                    (linear_attr,
                     f'{args.hess_path}/{idx}_{in_hess_name}_hin.pt',
                     f'{args.hess_path}/{idx}_{out_hess_name}_hout.pt'))

            with torch.enable_grad():
                finetune_decoder_layer(mixed_layer, attr_map, f'{idx}_{name}',
                                       device, train_dl, valid_dl, orig_dtype,
                                       position_embeddings, args)

        cb = cb.cpu()
        utils.clean()

    for quant_i, (linear_attr, name, in_hess_name, out_hess_name, rcp, skip_ft,
                  one_sided_override) in enumerate(quant_order):
        quant_linear = attrgetter(linear_attr)(mixed_layer)
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        data = torch.load(save_path)
        data['SU'] = quant_linear.SU.data.to(orig_dtype).cpu()
        data['SV'] = (
            quant_linear.SV.data /
            data['Wscale'].to(quant_linear.SV.device)).to(orig_dtype).cpu()

        if quant_linear.tlut is not None:
            data['tlut'] = quant_linear.tlut.data.to(orig_dtype).cpu()
        torch.save(data, save_path)

    mixed_layer = mixed_layer.to(orig_dtype).cpu()

    utils.clean()
    torch.set_grad_enabled(False)


def infer(args, end_dev, n_layers, in_q, out_q):
    with torch.no_grad():
        fake_dev_map = {
            'model.embed_tokens': 0,
            'model.rotary_emb': 0,
            'model.norm': max(end_dev - 1, 0),
            'lm_head': max(end_dev - 1, 0),
        }
        per_dev = math.ceil(n_layers / max(end_dev, 1))
        for i in range(n_layers):
            fake_dev_map[f'model.layers.{i}'] = i // per_dev

        model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     torch_dtype='auto',
                                                     device_map=fake_dev_map,
                                                     low_cpu_mem_usage=True)
        while True:
            data = in_q.get()
            if data is None:
                return
            out_q.put(
                model(data.to(0))['logits'][:, :-1].contiguous().softmax(
                    dim=-1).cpu())


def finetune_susv_e2e(quant_model, start_dev, devset, orig_dtype, args):

    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=infer,
                   args=(args, start_dev, len(quant_model.model.layers), in_q,
                         out_q))
    p.start()

    train_dl, valid_dl = utils.split_data(devset, devset, args)

    optim = torch.optim.Adam(quant_model.parameters(), lr=args.ft_lr)

    best_loss = utils.calculate_ce_loss_model(quant_model, valid_dl, start_dev,
                                              in_q, out_q)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_sd = copy.deepcopy(quant_model.state_dict())
    glog.info(f'initial loss {best_loss}')
    worse_ct = 0
    for epoch in range(args.ft_epochs):
        for bidx, (source, _) in enumerate(train_dl):
            in_q.put(source)
            with torch.autocast(device_type='cuda',
                                dtype=orig_dtype,
                                enabled=True):
                output = quant_model(
                    source.to(start_dev))['logits'][:, :-1].contiguous()
                target = out_q.get().to(output.device)
                target = target.view(-1, target.shape[-1])
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                             target)
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_ce_loss_model(quant_model, valid_dl,
                                                      start_dev, in_q, out_q)
            if test_loss < best_loss:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(quant_model.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    in_q.put(None)
    p.join()
    with torch.no_grad():
        quant_model.load_state_dict(best_sd)
