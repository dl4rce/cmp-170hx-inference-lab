#!/usr/bin/env python3
"""One real JPEG through the vision processor. Pass a local .jpg path."""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import add_common_args, stream_chat  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("jpeg", type=Path)
    args = p.parse_args()
    raw = args.jpeg.read_bytes()
    if raw[:3] != b"\xff\xd8\xff":
        raise SystemExit(f"{args.jpeg} is not a JPEG")
    b64 = base64.b64encode(raw).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What animal is in this photo? Answer with one English word only."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }
    ]
    r = stream_chat(
        args.base_url,
        args.model,
        messages,
        max_tokens=16,
        enable_thinking=False,
    )
    print(f"bytes={len(raw)} TTFT={r['ttft']:.3f}s total={r['total']:.3f}s prompt={r['prompt_tokens']}")
    print("VISION", r["text"].strip())
    print("JSONRESULT " + json.dumps({"ttft": r["ttft"], "total": r["total"], "text": r["text"].strip()}))


if __name__ == "__main__":
    main()
