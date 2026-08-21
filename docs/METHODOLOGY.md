# Methodology

## What ran

One NVIDIA CMP 170HX (GA100, 70 SMs, 64 GB HBM2e, 250 W). Driver 610.57.04. FMA unlocked. PCIe **Gen2 ×16** on this instance (width is not guaranteed on a random listing).

Engine: vLLM 0.27.1, PyTorch 2.13.0+cu130, `--kv-cache-dtype fp8`, `--gpu-memory-utilization 0.95`, `--enable-chunked-prefill`, prefix caching on.

Checkpoints (public Hugging Face):

- `philbert440/Qwen3.8-27B-W4A16-AWQ` (with MTP weights)
- `Qwen/Qwen2.5-72B-Instruct-AWQ`

Scripts in `scripts/` talk to a local OpenAI-compatible server (`--base-url http://127.0.0.1:8000` by default). They do not contain hostnames, IPs, or cloud account data.

## Baseline

Same 27B W4A16 recipe on a **2× RTX PRO 2000 16 GB** workstation, tensor-parallel 2 over PCIe, no NVLink, ~140 W pair. That box cannot load the 38.8 GiB 72B AWQ.

## Protocol notes

- Decode medians are three runs of 400 tokens with `ignore_eos` so length is fixed.
- Qwen3 thinking-on vs thinking-off changes tok/s a lot (88 vs 109 on 27B MTP-3). Production-comparable numbers in this repo are **thinking on**, unless a gate required thinking off (buried-key exact string, one-word vision).
- Buried-key prompts are fitted with `/tokenize`. Pass means the visible completion is exactly `K-7a3c91e0b24f58d1c6a90e2b`.
- Power is `nvidia-smi power.draw` sampled every 0.4 s during the run, not a lab-grade shunt.
- Restarting vLLM: killing the HTTP port is not enough. The `VLLM::EngineCore` compute process can keep ~60 GB allocated; the next start then fails `gpu_memory_utilization=0.95`.

## Re-run (generic)

```bash
# 27B MTP-3 example. Point --model at the local snapshot.
vllm serve /path/to/Qwen3.8-27B-W4A16-AWQ \
  --served-model-name qwen38-local \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

python3 scripts/bench_decode.py --base-url http://127.0.0.1:8000
python3 scripts/bench_mtp.py --base-url http://127.0.0.1:8000 --kv-tokens 1013443
python3 scripts/bench_mtp_depth.py --base-url http://127.0.0.1:8000 --label mtp3
python3 scripts/bench_power.py --base-url http://127.0.0.1:8000
python3 scripts/bench_buried_key.py --base-url http://127.0.0.1:8000 --target 110786
python3 scripts/bench_vision.py --base-url http://127.0.0.1:8000 /path/to/photo.jpg
```

72B: `max-model-len 32768`, no MTP, `--served-model-name qwen72-awq`. Then:

```bash
BASE_URL=http://127.0.0.1:8000 MODEL=qwen72-awq TOKENIZER=/path/to/Qwen2.5-72B-Instruct-AWQ \
  bash scripts/bench_vllm_random.sh
python3 scripts/bench_72b.py --base-url http://127.0.0.1:8000 --model qwen72-awq
```

Do not force 128K with `VLLM_ALLOW_LONG_MAX_MODEL_LEN`. Why this suite (vs ShareGPT / GuideLLM / MMLU): [`docs/70B.md`](70B.md).

256K: same 27B MTP-3 serve with `--max-model-len 262144`. Then `bench_buried_key.py --target 200000`.
