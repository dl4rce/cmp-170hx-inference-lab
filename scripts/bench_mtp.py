#!/usr/bin/env python3
"""MTP packing math plus 1..N parallel code-prompt decode."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import add_common_args, stream_chat  # noqa: E402

CODE = (
    "Write a complete Python module: thread-safe LRU cache with TTL, "
    "type hints, and docstrings. Count nothing, just emit code."
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--kv-tokens", type=int, required=True, help="GPU KV cache size from the vLLM log")
    p.add_argument("--agents", default="1,2,4,8,16")
    args = p.parse_args()
    user = [{"role": "user", "content": CODE}]

    print("--- packing (shared KV / per-agent context) ---")
    pack = []
    for ctx in (131072, 65536, 32768, 16384, 8192):
        n = args.kv_tokens // ctx
        pack.append({"ctx": ctx, "agents": n})
        print(f"  {ctx:>7,} tok/agent  ->  {n:3d} concurrent")

    print("--- single-agent code decode (400 tok, n=3) ---")
    d = []
    for i in range(3):
        r = stream_chat(args.base_url, args.model, user, 400, ignore_eos=True)
        d.append(r)
        print(f"  run {i+1}: {r['tps']:.2f} t/s  TTFT {r['ttft']*1000:.0f} ms")
    print(f"  median {statistics.median(x['tps'] for x in d):.2f} t/s")

    print("--- parallel agents (300 tok) ---")
    sweep = []
    for agents in [int(x) for x in args.agents.split(",")]:
        rows: list = []
        lock = threading.Lock()

        def work() -> None:
            r = stream_chat(args.base_url, args.model, user, 300, ignore_eos=True)
            with lock:
                rows.append(r)

        th = [threading.Thread(target=work) for _ in range(agents)]
        t0 = time.perf_counter()
        for x in th:
            x.start()
        for x in th:
            x.join()
        wall = time.perf_counter() - t0
        tot = sum(x["out"] for x in rows)
        row = {
            "agents": agents,
            "ok": len(rows),
            "agg": tot / wall,
            "per": statistics.median(x["tps"] for x in rows),
            "wall": wall,
        }
        sweep.append(row)
        print(
            f"  {agents:2d} agents  agg {row['agg']:7.2f} t/s | "
            f"per-agent {row['per']:6.2f} | {row['ok']}/{agents} ok"
        )

    print("JSONRESULT " + json.dumps({"pack": pack, "median_1": statistics.median(x["tps"] for x in d), "sweep": sweep}))


if __name__ == "__main__":
    main()
