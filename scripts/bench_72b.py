#!/usr/bin/env python3
"""72B W4A16 decode + 4-agent + ~30k buried key. Use --model qwen72-awq."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_buried_key import KEY, buried_messages, fit  # noqa: E402
from client import add_common_args, stream_chat  # noqa: E402

CODE = (
    "Write a complete Python module: thread-safe LRU cache with TTL, "
    "type hints, and docstrings. Count nothing, just emit code."
)


def smi() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,memory.used", "--format=csv,noheader"],
        text=True,
    ).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.set_defaults(model="qwen72-awq")
    args = p.parse_args()
    user = [{"role": "user", "content": CODE}]
    print("gpu", smi())
    runs = [
        stream_chat(args.base_url, args.model, user, 400, ignore_eos=True) for _ in range(3)
    ]
    for i, r in enumerate(runs, 1):
        print(f"  {i}: {r['tps']:.2f} t/s  TTFT {r['ttft']*1000:.0f} ms")
    print("median", round(statistics.median(x["tps"] for x in runs), 2))

    rows: list = []
    lock = threading.Lock()

    def work() -> None:
        r = stream_chat(args.base_url, args.model, user, 220, ignore_eos=True)
        with lock:
            rows.append(r)

    ths = [threading.Thread(target=work) for _ in range(4)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    tot = sum(x["out"] for x in rows)
    print(f"4 agents agg {tot/wall:.1f} per {statistics.median(x['tps'] for x in rows):.2f}")

    repeats, fitted = fit(args.base_url, args.model, 30000)
    print("buried fitted", repeats, fitted)
    b = stream_chat(
        args.base_url,
        args.model,
        buried_messages(repeats, 30000),
        max_tokens=32,
        enable_thinking=False,
    )
    print("buried PASS", b["text"].strip() == KEY, "prompt", b["prompt_tokens"], "ttft", round(b["ttft"], 2))
    print(
        "JSONRESULT "
        + json.dumps(
            {
                "median_tok_s": statistics.median(x["tps"] for x in runs),
                "agg_4": tot / wall,
                "buried_pass": b["text"].strip() == KEY,
                "buried_prompt": b["prompt_tokens"],
            }
        )
    )


if __name__ == "__main__":
    main()
