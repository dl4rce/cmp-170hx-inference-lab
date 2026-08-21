#!/usr/bin/env python3
"""Exact buried-key retrieval at a target prompt length. Thinking must be off."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import add_common_args, stream_chat, tokenize  # noqa: E402

FILLER = (
    "The repository contains independently deployed inference, caching, and "
    "observability services. Every change requires deterministic validation, "
    "boundary accounting, and measured end-to-end walltime. "
)
KEY = "K-7a3c91e0b24f58d1c6a90e2b"


def buried_messages(repeats: int, target: int) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a deterministic cache correctness probe. Follow the final instruction exactly.",
        },
        {
            "role": "user",
            "content": (
                f"CACHE_NAMESPACE=170hx-lab\nTARGET={target}\nBURIED_KEY={KEY}\n"
                f"<workspace_context>\n{FILLER * repeats}\n</workspace_context>\n"
                "Return only the exact BURIED_KEY value, with no punctuation or explanation."
            ),
        },
    ]


def fit(base: str, model: str, target: int) -> tuple[int, int]:
    low, high = 0, 200
    while tokenize(base, model, buried_messages(high, target)) < target:
        high *= 2
        if high > 40000:
            break
    best_r, best_n = 0, 0
    while low <= high:
        mid = (low + high) // 2
        n = tokenize(base, model, buried_messages(mid, target))
        if n <= target:
            best_r, best_n = mid, n
            low = mid + 1
        else:
            high = mid - 1
    return best_r, best_n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--target", type=int, default=110786)
    args = p.parse_args()
    repeats, fitted = fit(args.base_url, args.model, args.target)
    print(f"fitted repeats={repeats} tokenize={fitted}")
    r = stream_chat(
        args.base_url,
        args.model,
        buried_messages(repeats, args.target),
        max_tokens=32,
        enable_thinking=False,
    )
    ok = r["text"].strip() == KEY
    prefill = (r["prompt_tokens"] or 0) / max(r["ttft"], 1e-6)
    print(
        f"prompt={r['prompt_tokens']} TTFT={r['ttft']:.2f}s prefill={prefill:.1f} "
        f"PASS={ok} RAW={r['text'][:80]!r}"
    )
    print(
        "JSONRESULT "
        + json.dumps(
            {
                "prompt_tokens": r["prompt_tokens"],
                "ttft": r["ttft"],
                "prefill_tok_s": prefill,
                "pass": ok,
            }
        )
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
