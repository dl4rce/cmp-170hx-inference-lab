#!/usr/bin/env python3
"""Minimal OpenAI-compatible streaming client for the 170HX lab scripts."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="vLLM OpenAI base URL")
    p.add_argument("--model", default="qwen38-local")
    return p


def stream_chat(
    base_url: str,
    model: str,
    messages: list,
    max_tokens: int = 400,
    temperature: float = 0.0,
    ignore_eos: bool = False,
    enable_thinking: bool | None = None,
    timeout: int = 1800,
) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True
    if enable_thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    n = 0
    prompt_tok = None
    text: list[str] = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            ch = ev.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(piece)
            if ev.get("usage"):
                n = ev["usage"].get("completion_tokens") or n
                prompt_tok = ev["usage"].get("prompt_tokens")
    tot = time.perf_counter() - t0
    if ttft is None:
        ttft = tot
    return {
        "ttft": ttft,
        "total": tot,
        "out": n,
        "prompt_tokens": prompt_tok,
        "text": "".join(text),
        "tps": n / max(tot - ttft, 1e-6),
    }


def tokenize(base_url: str, model: str, messages: list) -> int:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/tokenize",
        data=json.dumps({"model": model, "messages": messages}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return int(d.get("count") or len(d.get("tokens") or []))
