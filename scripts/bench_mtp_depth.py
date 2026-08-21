#!/usr/bin/env python3
"""Three-run code-prompt decode. Restart vLLM with a different MTP depth between runs."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
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
    p.add_argument("--label", default="mtp")
    args = p.parse_args()
    runs = []
    for i in range(3):
        r = stream_chat(
            args.base_url,
            args.model,
            [{"role": "user", "content": CODE}],
            max_tokens=400,
            ignore_eos=True,
        )
        runs.append(r["tps"])
        print(f"{args.label} {i+1}: {r['tps']:.2f} t/s  TTFT {r['ttft']*1000:.0f} ms")
    print(f"{args.label} median {statistics.median(runs):.2f}")
    print("JSONRESULT " + json.dumps({"label": args.label, "runs": runs, "median": statistics.median(runs)}))


if __name__ == "__main__":
    main()
