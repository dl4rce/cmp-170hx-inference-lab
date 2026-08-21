#!/usr/bin/env python3
"""Compare local safetensors SHA-256 to Hugging Face LFS oids. No tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hub_lfs_oids(repo_id: str) -> dict[str, str]:
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main?recursive=1"
    req = urllib.request.Request(url, headers={"User-Agent": "cmp-170hx-lab"})
    with urllib.request.urlopen(req, timeout=120) as r:
        items = json.loads(r.read())
    if not isinstance(items, list):
        raise SystemExit(f"unexpected Hub tree payload: {type(items).__name__}")
    out: dict[str, str] = {}
    for it in items:
        if not isinstance(it, dict) or it.get("type") != "file":
            continue
        path = it.get("path") or ""
        if not path.endswith(".safetensors"):
            continue
        lfs = it.get("lfs") or {}
        oid = str(lfs.get("oid") or "")
        if oid.startswith("sha256:"):
            oid = oid.split(":", 1)[1]
        if len(oid) == 64:
            out[path] = oid
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="org/name on the Hub")
    p.add_argument("--dir", required=True, type=Path)
    args = p.parse_args()
    expected = hub_lfs_oids(args.repo)
    if not expected:
        print("no LFS safetensors listed on Hub", file=sys.stderr)
        return 2
    failed = 0
    for rel, want in sorted(expected.items()):
        local = args.dir / rel
        if not local.is_file():
            print(f"MISSING {rel}")
            failed += 1
            continue
        got = sha256_file(local)
        ok = got == want
        print(f"{'OK' if ok else 'MISMATCH'} {rel}")
        if not ok:
            print(f"  hub  {want}")
            print(f"  disk {got}")
            failed += 1
    print(f"checked {len(expected)} shards, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
