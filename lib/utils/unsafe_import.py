# functions in this file cause circular imports so they cannot be loaded into __init__

import json
import os

import accelerate
import torch
import transformers

from model.llama import LlamaForCausalLM
from model.llama4 import Llama4ForCausalLM
from model.llama4_orig import Llama4ForCausalLM as Llama4ForCausalLMOrig


def model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):

    # AutoConfig fails to read name_or_path correctly
    bad_config = transformers.AutoConfig.from_pretrained(path)
    is_quantized = hasattr(bad_config, 'quip_params')
    model_type = bad_config.model_type

    if is_quantized:
        if model_type == 'llama':
            model_str = transformers.LlamaConfig.from_pretrained(
                path)._name_or_path
            model_cls = LlamaForCausalLM
        elif model_type.startswith('llama4'):
            model_str = transformers.LlamaConfig.from_pretrained(
                path)._name_or_path
            model_cls = Llama4ForCausalLM
        else:
            raise Exception
    else:
        if model_type.startswith('llama4'):
            model_str = transformers.LlamaConfig.from_pretrained(
                path)._name_or_path
            model_cls = Llama4ForCausalLMOrig
        else:
            model_cls = transformers.AutoModelForCausalLM
            model_str = path

    print(model_cls)
    if device_map is None:
        mmap = {
            i: f"{torch.cuda.mem_get_info(i)[1]*max_mem_ratio/(1 << 30)}GiB"
            for i in range(torch.cuda.device_count())
        }
        model = model_cls.from_pretrained(path,
                                          torch_dtype='auto',
                                          low_cpu_mem_usage=True,
                                          attn_implementation='sdpa')
        device_map = accelerate.infer_auto_device_map(
            model,
            no_split_module_classes=[
                'LlamaDecoderLayer', 'Llama4TextDecoderLayer'
            ],
            max_memory=mmap)
    model = model_cls.from_pretrained(path,
                                      torch_dtype='auto',
                                      low_cpu_mem_usage=True,
                                      attn_implementation='sdpa',
                                      device_map=device_map)

    return model, model_str
