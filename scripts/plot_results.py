#!/usr/bin/env python3
"""Render lab JSON to SVG figures. Optional PNG if matplotlib is installed."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "lab-2026-08-21.json").read_text())
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INK = "#1b1f24"
MUTED = "#5c6570"
ACCENT = "#c45c4a"
BAR = "#2f4f6f"
PAPER = "#f7f5f1"


def svg_bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str, highlight: int | None = None) -> None:
    w, h = 840, 420
    left, right, top, bottom = 70, 30, 56, 70
    plot_w, plot_h = w - left - right, h - top - bottom
    vmax = max(values) * 1.15
    n = len(values)
    gap = 18
    bw = (plot_w - gap * (n + 1)) / n
    bars = []
    texts = []
    for i, (lab, val) in enumerate(zip(labels, values)):
        x = left + gap + i * (bw + gap)
        bh = val / vmax * plot_h
        y = top + plot_h - bh
        fill = ACCENT if i == highlight else BAR
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{fill}"/>')
        texts.append(
            f'<text x="{x + bw/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="13" fill="{INK}" font-family="ui-sans-serif, system-ui, sans-serif">{val:g}</text>'
        )
        texts.append(
            f'<text x="{x + bw/2:.1f}" y="{h - 28:.1f}" text-anchor="middle" font-size="13" fill="{MUTED}" font-family="ui-sans-serif, system-ui, sans-serif">{lab}</text>'
        )
    yticks = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        yy = top + plot_h * (1 - frac)
        v = vmax * frac
        yticks.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{w-right}" y2="{yy:.1f}" stroke="#e6e2db"/>')
        yticks.append(
            f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}" font-family="ui-sans-serif, system-ui, sans-serif">{v:.0f}</text>'
        )
    body = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="{PAPER}"/>
  <text x="{left}" y="28" font-size="18" font-weight="600" fill="{INK}" font-family="ui-sans-serif, system-ui, sans-serif">{title}</text>
  <text x="{left}" y="46" font-size="12" fill="{MUTED}" font-family="ui-sans-serif, system-ui, sans-serif">{ylabel} · CMP 170HX lab 2026-08-21 · vLLM 0.27.1</text>
  {''.join(yticks)}
  {''.join(bars)}
  {''.join(texts)}
</svg>
'''
    path.write_text(body)


def main() -> None:
    depths = DATA["qwen38_mtp"]["depths"]
    svg_bar(
        OUT / "mtp-depth.svg",
        "Qwen3.8 W4A16 code decode vs MTP draft depth",
        [f"MTP-{d['n']}" if d["n"] else "off" for d in depths],
        [d["code_tok_s"] for d in depths],
        "median tok/s (400 tokens, thinking on, n=3)",
        highlight=3,
    )
    sweep = DATA["qwen38_mtp"]["mtp3"]["sweep_agg_tok_s"]
    keys = sorted(sweep, key=int)
    svg_bar(
        OUT / "concurrency.svg",
        "MTP-3 aggregate tok/s vs parallel agents",
        [f"{k} agent" if k == "1" else f"{k} agents" for k in keys],
        [sweep[k] for k in keys],
        "aggregate tok/s, short code prompt",
        highlight=4,
    )
    pw = DATA["power_w"]
    svg_bar(
        OUT / "power.svg",
        "Board power on 27B MTP-3",
        ["idle", "1 agent", "16 agents", "72B 1 agent"],
        [
            pw["idle"],
            pw["qwen38_mtp3_1_agent_median"],
            pw["qwen38_mtp3_16_agent_median"],
            pw["qwen72_1_agent_mean"],
        ],
        "watts (nvidia-smi power.draw)",
        highlight=2,
    )
    svg_bar(
        OUT / "compare-27b.svg",
        "27B W4A16 single-stream decode",
        ["2x 2000 off", "2x 2000 MTP-3", "170HX off", "170HX MTP-3"],
        [
            DATA["baseline"]["qwen38_no_mtp_tok_s"],
            DATA["baseline"]["qwen38_mtp3_tok_s"],
            DATA["qwen38_no_mtp"]["decode_tok_s"],
            DATA["qwen38_mtp"]["mtp3"]["single_tok_s"],
        ],
        "tok/s · same checkpoint class, TP=2 on the dual-16GB box",
        highlight=3,
    )
    cells = json.loads((ROOT / "results" / "vllm-random-72b.json").read_text())["cells"]
    c1 = [c for c in cells if c["concurrency"] == 1 and c["osl"] == 128]
    svg_bar(
        OUT / "72b-isl-osl.svg",
        "Qwen2.5-72B AWQ  —  vLLM random ISL/OSL, conc=1",
        [f"{c['isl']}→{c['osl']}" for c in c1],
        [c["output_tok_s"] for c in c1],
        "output tok/s (generated / wall) · ignore_eos",
        highlight=0,
    )
    svg_bar(
        OUT / "72b-vs-27b.svg",
        "Single card: 27B MTP-3 vs 72B AWQ",
        ["27B MTP-3", "72B code", "72B 512→128", "72B 4-wide"],
        [
            DATA["qwen38_mtp"]["mtp3"]["single_tok_s"],
            DATA["qwen72_awq"]["decode_tok_s"],
            23.19,
            79.37,
        ],
        "tok/s  ·  4-wide is aggregate output throughput",
        highlight=1,
    )
    print("wrote", *sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
