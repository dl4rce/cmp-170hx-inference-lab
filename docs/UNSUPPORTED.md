# Models that did not serve on the frozen lab recipe

The frozen stack behind the Qwen numbers is **vLLM 0.27.1**, transformers 5.15.1, one unlocked CMP 170HX (cc **8.0**), FP8 KV.

Two models failed on it. **Both were later served on the same card** with a newer vLLM in a separate venv — see the README section “Muse Glimmer 30B and Gemma 4 31B”. This page keeps the failure modes, because they are the useful part.

## Gemma 4 31B — the AWQ repack (`QuantTrio/gemma-4-31B-it-AWQ`)

Downloaded (~20 GiB). **7/7** `.safetensors` matched Hub LFS SHA-256.

Did not serve:

1. `Gemma4ForConditionalGeneration` + transformers 5.15 per-layer `head_dim` → `AmbiguousGlobalPerLayerAttributeError` in vLLM’s config convertor.
2. A local `allow_global_per_layer_attribute_access=true` on `config.json` got past that. `--kv-cache-dtype fp8` then failed: Triton attention on this SKU has no native FP8 KV (`SM89+`).
3. `--kv-cache-dtype bfloat16`: `AssertionError` loading `torch.Size([512])` into `torch.Size([256])` in `vllm/.../gemma4.py`.

Item 3 is a checkpoint problem, not a GPU problem: Gemma 4 uses different head dims for sliding vs global layers, and this repack does not carry them the way vLLM expects. The official **QAT compressed-tensors** build (`google/gemma-4-31B-it-qat-w4a16-ct`, one shard, hash-checked) loads without any config editing. Weights for the AWQ repack were deleted after the failed load.

## Muse Glimmer 30B on vLLM 0.27.1

Hub `config.json` architectures: `MuseGlimmerForConditionalGeneration` (`model_type: muse_glimmer`, native 131072).

vLLM 0.27.1’s generation registry on this box listed `Gemma4*` and `Qwen3_5Moe*` — **no Muse/Glimmer class**. Nothing to configure around; the model support simply is not in that release. A newer build has `MuseGlimmerForCausalLM`, `MuseGlimmerForConditionalGeneration` and the `muse_glimmer` tool/reasoning parsers, and serves the W4A16 checkpoint at TP=1 in 21 GiB.

NVFP4 siblings still do not apply here — those need Blackwell. W4A16 is the Ampere-legal quant.

## Two things that bite every non-Qwen model on cc 8.0

Independent of the model:

- **FP8 KV cache is unavailable.** Triton attention raises `FP8 KV cache is not supported ... native FP8 (fp8e4nv) requires SM89+`. Use `--kv-cache-dtype bfloat16` and expect roughly half the KV pool.
- **FlashInfer JIT may not build.** On this image `nvcc` (CUDA 13.3) and the bundled cccl headers disagree — `"CUDA compiler and CUDA toolkit headers are incompatible"` — killing both the attention and sampling kernels. `--attention-backend TRITON_ATTN` plus `VLLM_USE_FLASHINFER_SAMPLER=0` avoids it.
