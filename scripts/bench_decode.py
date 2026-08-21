#!/usr/bin/env python3
"""Single-stream TTFT, decode, 2/4-agent aggregate, long-context prefill."""
from __future__ import annotations

import json
import statistics
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import add_common_args, stream_chat  # noqa: E402

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    args = p.parse_args()
    user = lambda text: [{"role": "user", "content": text}]

    t = [
        stream_chat(args.base_url, args.model, user("Hi"), max_tokens=8)["ttft"]
        for _ in range(7)
    ]
    print(f"TTFT short prompt      : {statistics.median(t)*1000:7.1f} ms  (median of 7)")

    d = []
    for _ in range(3):
        r = stream_chat(
            args.base_url,
            args.model,
            user("Count slowly from 1 to 200."),
            max_tokens=400,
            ignore_eos=True,
        )
        d.append(r["tps"])
    print(f"Single-agent decode    : {statistics.median(d):7.2f} tok/s (median of 3)")

    long_p = "Write a very detailed technical text about TCP congestion control."
    out_res = {"ttft_ms": statistics.median(t) * 1000, "decode_tok_s": statistics.median(d)}
    for agents in (2, 4):
        rows: list = []
        lock = threading.Lock()

        def work() -> None:
            r = stream_chat(
                args.base_url, args.model, user(long_p), max_tokens=300, ignore_eos=True
            )
            with lock:
                rows.append(r)

        th = [threading.Thread(target=work) for _ in range(agents)]
        t0 = __import__("time").perf_counter()
        for x in th:
            x.start()
        for x in th:
            x.join()
        wall = __import__("time").perf_counter() - t0
        tot = sum(x["out"] for x in rows)
        print(
            f"{agents} agents aggregate     : {tot/wall:7.2f} tok/s | "
            f"per-agent {statistics.median(x['tps'] for x in rows):6.2f} tok/s"
        )
        out_res[f"agg_{agents}"] = tot / wall

    words = (
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua. "
    )
    big = words * 2400
    r = stream_chat(
        args.base_url,
        args.model,
        user(big + "\n\nSummarise in one sentence."),
        max_tokens=60,
    )
    print(f"Long-context prefill   : TTFT {r['ttft']:7.2f} s  prompt_tok={r['prompt_tokens']}")
    out_res["prefill_ttft_s"] = r["ttft"]
    print("JSONRESULT " + json.dumps(out_res))


if __name__ == "__main__":
    main()
