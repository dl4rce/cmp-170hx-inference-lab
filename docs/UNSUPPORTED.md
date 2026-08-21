# Models that did not serve on this lab recipe

Same frozen stack as the numbers: **vLLM 0.27.1**, transformers 5.15.1, one unlocked CMP 170HX (cc **8.0**). We did **not** upgrade the engine to chase newer arches.

## Gemma 4 31B (`QuantTrio/gemma-4-31B-it-AWQ`)

Downloaded (~20 GiB). **7/7** `.safetensors` matched Hub LFS SHA-256.

Did not serve:

1. `Gemma4ForConditionalGeneration` + transformers 5.15 per-layer `head_dim` → `AmbiguousGlobalPerLayerAttributeError` in vLLM’s config convertor.
2. A local `allow_global_per_layer_attribute_access=true` on `config.json` got past that. `--kv-cache-dtype fp8` then failed: Triton attention on this SKU has no native FP8 KV (`SM89+`).
3. `--kv-cache-dtype bfloat16`: `AssertionError` loading `torch.Size([512])` into `torch.Size([256])` in `vllm/.../gemma4.py`.

Weights were deleted after the failed load to free overlay space. Re-run would need a newer vLLM, not this recipe.

## Muse Glimmer 30B (`RedHatAI/Muse-Glimmer-30B-W4A16`)

Hub `config.json` architectures: `MuseGlimmerForConditionalGeneration` (`model_type: muse_glimmer`, native 131072).

vLLM 0.27.1’s generation registry on this box listed `Gemma4*` and `Qwen3_5Moe*` — **no Muse/Glimmer class**. Public deploy docs use `vllm/vllm-openai:muse-glimmer`. NVFP4 siblings need Blackwell. The 170HX is Ampere cc 8.0.

Weights were **not** downloaded (~20 GiB that this engine cannot load).
