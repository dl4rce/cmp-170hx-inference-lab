# Methodology

## What ran

One NVIDIA CMP 170HX (GA100, 70 SMs, 64 GB HBM2e, 250 W). Driver 610.57.04. FMA unlocked. PCIe **Gen2 ×16** on this instance (width is not guaranteed on a random listing).

**Two engines**, deliberately kept in separate venvs on the same box.

**Engine A — the frozen recipe** behind every Qwen number: vLLM 0.27.1, PyTorch 2.13.0+cu130, `--kv-cache-dtype fp8`, `--gpu-memory-utilization 0.95`, `--enable-chunked-prefill`, prefix caching on.

- `philbert440/Qwen3.8-27B-W4A16-AWQ` (with MTP weights)
- `Qwen/Qwen2.5-72B-Instruct-AWQ`
- `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` (6/6 shards SHA-256 vs Hub LFS)

**Engine B — a dev-branch vLLM** (`0.1.dev19962+gc4dc58557`, transformers 5.15.1) installed into a second venv because 0.27.1 has no Muse Glimmer class and mishandles Gemma 4's per-layer attention config. It runs with **`--kv-cache-dtype bfloat16`**, `--attention-backend TRITON_ATTN`, `VLLM_USE_FLASHINFER_SAMPLER=0`.

- `dudeman2512/Muse-Glimmer-30B-INT4-W4A16` (5/5 shards SHA-256 vs Hub LFS)
- `google/gemma-4-31B-it-qat-w4a16-ct` (1/1 shard SHA-256 vs Hub LFS)

The split is the point: upgrading in place would have invalidated the published 0.27.1 numbers. **Engine A and Engine B results are not directly comparable** — FP8 vs bf16 KV alone roughly halves the pool. Failure modes for both models on Engine A, and the two cc 8.0 constraints that apply to any non-Qwen model here: [`UNSUPPORTED.md`](UNSUPPORTED.md).

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

Qwen3.6 35B-A3B (no MTP, this engine’s fast path):

```bash
vllm serve /path/to/Qwen3.6-35B-A3B-AWQ-4bit \
  --served-model-name qwen36-local \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code

python3 scripts/verify_hf_dir.py --repo cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit --dir /path/to/snapshot
python3 scripts/bench_decode.py --base-url http://127.0.0.1:8000 --model qwen36-local
python3 scripts/bench_mtp.py --base-url http://127.0.0.1:8000 --model qwen36-local --kv-tokens 3159025
```

## Re-run: Muse Glimmer 30B and Gemma 4 31B (Engine B)

Install the dev-branch vLLM into its **own** venv so the 0.27.1 numbers above stay reproducible:

```bash
python3 -m venv /path/to/muse-venv
VLLM_USE_PRECOMPILED=1 /path/to/muse-venv/bin/pip install \
  "vllm @ git+https://github.com/vllm-project/vllm.git@main"
```

**If that 404s**, `main` has moved ahead of the last published wheel — `VLLM_USE_PRECOMPILED` needs a commit that has one, or it tries to compile CUDA from source. Find the newest commit that does, and install from a full clone so the checkout can resolve it:

```bash
git clone --filter=blob:none https://github.com/vllm-project/vllm.git /tmp/vllmsrc
cd /tmp/vllmsrc
for c in $(git log -40 --format=%H); do
  curl -sfI "https://wheels.vllm.ai/${c}/vllm/index.html" >/dev/null && { echo "$c"; break; }
done
# then, with that commit:
git checkout -q <commit>
export VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=<commit>
/path/to/muse-venv/bin/pip install /tmp/vllmsrc
```

Takes about 3 minutes. Two traps: create the venv **without** `--system-site-packages`, or a stock 0.27.1 in the parent environment silently satisfies the requirement and you end up with no Muse class; and `pip install git+...@<short-hash>` fails because the shallow clone cannot resolve it. Verify with:

```bash
/path/to/muse-venv/bin/python -c "from vllm.model_executor.models.registry import ModelRegistry as R; print('MuseGlimmerForConditionalGeneration' in R.get_supported_archs())"
```

The three environment variables matter as much as the flags:

```bash
VENV=/path/to/muse-venv
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0   # FlashInfer JIT cannot build sm_80 here

"$VENV/bin/vllm" serve /path/to/Muse-Glimmer-30B-INT4-W4A16 \
  --served-model-name muse-glimmer \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype bfloat16 \
  --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto

python3 scripts/bench_decode.py --base-url http://127.0.0.1:8000 --model muse-glimmer
python3 scripts/bench_power.py  --base-url http://127.0.0.1:8000 --model muse-glimmer --agents 16
```

Gemma 4 is the same serve line with `--reasoning-parser gemma4`, `--tool-call-parser gemma4`, and no `--generation-config`. Two traps:

- **`--kv-cache-dtype fp8` is not optional to change.** Triton attention rejects it on cc 8.0 (`native FP8 (fp8e4nv) requires SM89+`). Swapping to `--attention-backend FLASHINFER` to keep FP8 does **not** help — the JIT then fails on incompatible CUDA headers.
- **Query Gemma through `/v1/chat/completions`.** Raw `/v1/completions` on this instruction/thinking checkpoint degenerates into repeated tokens. Also start it at `--max-model-len 131072`: the long-context stage of `bench_decode.py` sends ~52K tokens and returns HTTP 400 against a 32K server.

## Dual-card runs

A second rental instance with **2× CMP 170HX** (128 GB, both Gen2 ×16, `PHB`, single NUMA node), same vLLM 0.27.1 / torch 2.13.0+cu130 / FP8 KV recipe. TP=1 there reproduced the single-card lab to within 0.1% (57.47 vs 57.52 tok/s, KV 1.194M vs 1.204M), which is the cross-instance sanity check for every number in this repo.

Interconnect first, before any model:

```bash
nvidia-smi topo -m            # PHB = both cards traverse the host bridge
nvidia-smi topo -p2p rw       # GNS = P2P fused off in the SKU
python3 -c "import torch; print(torch.cuda.can_device_access_peer(0,1))"
```

Then the same serve line twice, changing only `--tensor-parallel-size 1` → `2`, and a sharded comparison:

```bash
# one server per card, no collectives between them
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i vllm serve /path/to/Qwen3.8-27B-W4A16-AWQ \
    --served-model-name qwen38 --host 127.0.0.1 --port $((8000+i)) \
    --tensor-parallel-size 1 --max-model-len 131072 --max-num-seqs 16 \
    --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 &
done
python3 scripts/bench_sharded.py   # splits N agents across both ports
```

The 16-agent TP=2 figure was repeated three times and landed within 0.3% (413.6 / 414.5 / 414.5 tok/s).

Two script notes: `bench_power.py` originally parsed a single `nvidia-smi` row and crashed on a two-GPU box; it now sums power and memory across visible GPUs, takes max temperature, and reports `n_gpu`. Dual-card power figures are therefore **box totals**, and the TP=1 numbers include ~40 W from the idle second card. Also, on that image a preinstalled `torchaudio` (cu128) aborted every vLLM import against torch 2.13/cu130 — uninstalling it is harmless for text models.

## Coverage and what these numbers do not claim

Deliberate gaps, so nobody reads more into the table than is there:

- **Throughput, not quality.** No MMLU/GSM8K/perplexity anywhere in this repo. A 4-bit quant that got dumber would still post the same tok/s.
- **Muse and Gemma have no accuracy check.** The buried-key needle test was run on Qwen only. Both new models got a single greedy sanity prompt (`The capital of France is` → ` Paris.`) and nothing more.
- **Gemma's 128K window is nominal.** With bf16 KV the pool holds ~2.7 full contexts; behaviour near the window edge is untested.
- **Engine B has no MTP/speculative sweep.** Neither checkpoint was tested with a draft head.
- **Baseline is not a same-day rerun.** The dual-16GB numbers come from the same recipe on the owned box, quoted from earlier runs.
- **Power is `nvidia-smi`, not a shunt**, and one rental instance is one sample of silicon and cooling.
- **The dual-card conclusion is one dense 27B checkpoint.** An MoE, or a model too large for one card, moves the TP=1/TP=2 crossover. The P2P finding itself (`GNS`) is a property of the SKU and generalises; the exact 32% penalty does not.
