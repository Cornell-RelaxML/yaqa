To collect Hessians for Llama 1, 2, and 3 models, run the `get_hess_llama.py` script with `torchrun`. 
Parameters with `(WHATEVER FITS)` will need to be tuned by you depending on the size of your machine. 
As a guide, we were able to fit models under ~20B parameters on a single 8x80G node and 70B across 2 8x80G nodes (layers 0 through 39 on one, 40 through 79 on another).
This script processes layers independently so if you are on a cluster with a shared filesystem you can launch jobs in parallel across subsets of layers.
We do not recommend using `cpu_offload` unless it is faster to move things to CPU on your machine than recomputing gradients.
Accumulating in FP64 is not necessary but may give slight improvements in quantization performance. 
The actual Hessian collection computation still happens in FP32 *per sample*, but if for some reason your model requires FP64 you may also want to change [the computation](https://github.com/tsengalb99/qtip_2hess/blob/d627a60eb408c35e79b762a52455faa3e082c70a/hessian_llama/custom_linear_B.py#L61) to FP64 as well.
We recommend using Sketch B if you can afford it.

## Sketch A
```
torchrun --standalone --nproc-per-node=8 get_hess_llama.py \
    --save_path PATH \
    --orig_model HF_MODEL \
    --batch_size (WHATEVER FITS) \
    --start_layer (WHATEVER FITS) \
    --end_layer (WHATEVER FITS) \
    --hessian_sketch A \
    --power_iters 6 \
    --ctx_size 8192 \
    --n_seqs 4096 \
    (OPTIONAL)
    --fp64_accum (ACCUMULATE IN FP64) \
    --cpu_offload (CPU OFFLOAD, USUALLY SLOWER THAN SPLITTING BY start_layer/end_layer)
```

## Sketch B (Recommended)

```
torchrun --standalone --nproc-per-node=8 get_hess_llama.py \
    --save_path PATH \
    --orig_model HF_MODEL \
    --batch_size (WHATEVER FITS) \
    --start_layer (WHATEVER FITS) \
    --end_layer (WHATEVER FITS) \
    --hessian_sketch B \
    --power_iters 1 \
    --ctx_size 2048 \
    --n_seqs 65536 \
    (OPTIONAL)
    --fp64_accum (ACCUMULATE IN FP64) \
    --cpu_offload (CPU OFFLOAD, USUALLY SLOWER THAN SPLITTING BY start_layer/end_layer)
```
