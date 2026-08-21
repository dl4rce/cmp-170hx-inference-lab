# Scripts

All talk to a local OpenAI-compatible server. Defaults: `--base-url http://127.0.0.1:8000 --model qwen38-local`.

| Script | What it measures |
|---|---|
| `bench_decode.py` | TTFT, single decode, 2/4-agent aggregate, long prefill |
| `bench_mtp.py` | KV packing table + 1..N parallel code decode |
| `bench_mtp_depth.py` | Three-run code decode (restart vLLM with another `num_speculative_tokens` between labels) |
| `bench_buried_key.py` | Exact needle at `--target` tokens (`enable_thinking=false`) |
| `bench_vision.py` | One local JPEG via `data:image/jpeg;base64` |
| `bench_power.py` | `nvidia-smi power.draw` during 1-agent and N-agent decode |
| `bench_72b.py` | 72B decode, 4-agent, ~30k buried key (`--model qwen72-awq`) |
| `bench_vllm_random.sh` | Upstream `vllm bench serve --dataset-name random` ISL/OSL matrix |
| `plot_results.py` | Rebuild `figures/*.svg` from `results/*.json` |
