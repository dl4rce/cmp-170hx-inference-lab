#!/usr/bin/env python3
"""Sample nvidia-smi power.draw during 1-agent and N-agent decode."""
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
from client import add_common_args, stream_chat  # noqa: E402

CODE = (
    "Write a complete Python module: thread-safe LRU cache with TTL, "
    "type hints, and docstrings. Count nothing, just emit code."
)


def smi() -> dict:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=power.draw,temperature.gpu,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = [[float(x) for x in ln.split(",")] for ln in out.splitlines() if ln.strip()]
    # sum power/memory across all visible GPUs; max temp; mean utilisation
    p = sum(r[0] for r in rows)
    t = max(r[1] for r in rows)
    u = sum(r[2] for r in rows) / len(rows)
    m = sum(r[3] for r in rows)
    return {"power_w": p, "temp_c": t, "util": u, "mem_mib": m, "n_gpu": len(rows)}


def with_power(fn):
    stop = threading.Event()
    bucket: list = []

    def loop() -> None:
        while not stop.is_set():
            try:
                bucket.append(smi())
            except Exception:
                pass
            time.sleep(0.4)

    idle = smi()
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    try:
        result = fn()
    finally:
        stop.set()
        th.join(timeout=2)
    watts = [x["power_w"] for x in bucket] or [idle["power_w"]]
    result["power"] = {
        "idle_w": idle["power_w"],
        "mean_w": sum(watts) / len(watts),
        "max_w": max(watts),
        "median_w": statistics.median(watts),
        "max_temp_c": max((x["temp_c"] for x in bucket), default=idle["temp_c"]),
        "n_samples": len(watts),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--agents", type=int, default=16)
    args = p.parse_args()
    user = [{"role": "user", "content": CODE}]

    def one():
        return stream_chat(args.base_url, args.model, user, 300, ignore_eos=True)

    def many():
        rows: list = []
        lock = threading.Lock()

        def work() -> None:
            r = stream_chat(args.base_url, args.model, user, 220, ignore_eos=True)
            with lock:
                rows.append(r)

        ths = [threading.Thread(target=work) for _ in range(args.agents)]
        t0 = time.perf_counter()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.perf_counter() - t0
        tot = sum(x["out"] for x in rows)
        return {"ok": len(rows), "agg": tot / wall, "per": statistics.median(x["tps"] for x in rows), "wall": wall}

    print("idle", smi())
    one_r = with_power(one)
    print("1 agent", {k: one_r[k] for k in ("tps", "ttft") if k in one_r}, one_r["power"])
    n_r = with_power(many)
    print(f"{args.agents} agents", {k: n_r[k] for k in ("ok", "agg", "per", "wall")}, n_r["power"])
    print("JSONRESULT " + json.dumps({"one": one_r["power"], "n": n_r["power"], "n_agents": args.agents}))


if __name__ == "__main__":
    main()
