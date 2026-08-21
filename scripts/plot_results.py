#!/usr/bin/env python3
"""Render lab JSON to SVG figures. Optional PNG if matplotlib is installed."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "lab-2026-08-21.json").read_text())
DUAL = json.loads((ROOT / "results" / "dual-card-2026-08-21.json").read_text())
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INK = "#1b1f24"
MUTED = "#5c6570"
ACCENT = "#c45c4a"
BAR = "#2f4f6f"
GREEN = "#4a7c59"
PAPER = "#f7f5f1"
PALETTE = [BAR, ACCENT, GREEN, "#8a6d3b"]


def svg_bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str, highlight: int | None = None) -> None:
    w, h = 840, 420
    left, right, top, bottom = 70, 30, 56, 70
    plot_w, plot_h = w - left - right, h - top - bottom
    vmax = max(values) * 1.15 if values else 1
    n = len(values)
    gap = 18
    bw = (plot_w - gap * (n + 1)) / max(n, 1)
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
    path.write_text(_wrap(w, h, left, title, ylabel, yticks, bars, texts))


def svg_grouped(
    path: Path,
    title: str,
    groups: list[str],
    series: list[tuple[str, list[float]]],
    ylabel: str,
) -> None:
    w, h = 900, 460
    left, right, top, bottom = 70, 24, 56, 96
    plot_w, plot_h = w - left - right, h - top - bottom
    flat = [v for _, vals in series for v in vals]
    vmax = max(flat) * 1.18 if flat else 1
    n = len(groups)
    s = len(series)
    gap = 22
    gw = (plot_w - gap * (n + 1)) / max(n, 1)
    bw = gw / max(s, 1) * 0.86
    inner = (gw - bw * s) / 2
    bars, texts, legend = [], [], []
    for gi, lab in enumerate(groups):
        gx = left + gap + gi * (gw + gap)
        for si, (_, vals) in enumerate(series):
            val = vals[gi]
            x = gx + inner + si * bw
            bh = val / vmax * plot_h
            y = top + plot_h - bh
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{PALETTE[si % len(PALETTE)]}"/>'
            )
            if bh > 18:
                texts.append(
                    f'<text x="{x + bw/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="11" fill="{INK}" font-family="ui-sans-serif, system-ui, sans-serif">{val:g}</text>'
                )
        texts.append(
            f'<text x="{gx + gw/2:.1f}" y="{h - 52:.1f}" text-anchor="middle" font-size="13" fill="{MUTED}" font-family="ui-sans-serif, system-ui, sans-serif">{lab}</text>'
        )
    for si, (name, _) in enumerate(series):
        lx = left + si * 210
        legend.append(
            f'<rect x="{lx}" y="{h - 28}" width="12" height="12" fill="{PALETTE[si % len(PALETTE)]}"/>'
            f'<text x="{lx + 18}" y="{h - 17}" font-size="12" fill="{INK}" font-family="ui-sans-serif, system-ui, sans-serif">{name}</text>'
        )
    yticks = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        yy = top + plot_h * (1 - frac)
        v = vmax * frac
        yticks.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{w-right}" y2="{yy:.1f}" stroke="#e6e2db"/>')
        yticks.append(
            f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}" font-family="ui-sans-serif, system-ui, sans-serif">{v:.0f}</text>'
        )
    path.write_text(_wrap(w, h, left, title, ylabel, yticks, bars, texts + legend))


def _wrap(w: int, h: int, left: int, title: str, ylabel: str, yticks, bars, texts) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="{PAPER}"/>
  <text x="{left}" y="28" font-size="18" font-weight="600" fill="{INK}" font-family="ui-sans-serif, system-ui, sans-serif">{title}</text>
  <text x="{left}" y="46" font-size="12" fill="{MUTED}" font-family="ui-sans-serif, system-ui, sans-serif">{ylabel} · CMP 170HX lab 2026-08-21 · vLLM 0.27.1</text>
  {''.join(yticks)}
  {''.join(bars)}
  {''.join(texts)}
</svg>
'''


def _pack_vals(obj: dict) -> list[int]:
    p = obj["packing_agents"]
    return [p[k] for k in ("131072", "65536", "32768", "16384", "8192")]


def _sweep_vals(obj: dict) -> list[float]:
    s = obj["sweep_agg_tok_s"]
    return [round(s[k], 1) for k in ("1", "2", "4", "8", "16")]


def _muse_sweep(obj: dict) -> list[float]:
    keys = ("decode_tok_s", "agg_2_tok_s", "agg_4_tok_s", "agg_8_tok_s", "agg_16_tok_s")
    return [round(obj[k], 1) for k in keys]


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
        "27B MTP-3 aggregate tok/s vs parallel agents",
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

    q36 = DATA["qwen36_no_mtp"]
    q36m = DATA["qwen36_mtp"]
    svg_bar(
        OUT / "single-stream.svg",
        "Single-stream decode, sorted",
        ["35B-A3B", "35B MTP-3", "27B MTP-3", "Muse 30B", "27B off", "Gemma4 31B", "72B"],
        [
            round(q36["decode_tok_s"], 1),
            round(q36m["decode_tok_s"], 1),
            DATA["qwen38_mtp"]["mtp3"]["single_tok_s"],
            round(DATA["muse_glimmer_30b"]["decode_tok_s"], 1),
            DATA["qwen38_no_mtp"]["decode_tok_s"],
            round(DATA["gemma4_31b_qat"]["decode_tok_s"], 1),
            DATA["qwen72_awq"]["decode_tok_s"],
        ],
        "tok/s · one CMP 170HX (Muse/Gemma on bf16 KV)",
        highlight=0,
    )
    svg_bar(
        OUT / "kv-128k.svg",
        "KV pool at 128K (72B is native 32K)",
        ["35B-A3B", "Muse 30B*", "35B MTP-3", "27B off", "27B MTP-3", "Gemma4*", "72B@32K"],
        [
            round(q36["kv_tokens"] / 1e6, 2),
            round(DATA["muse_glimmer_30b"]["kv_tokens"] / 1e6, 2),
            round(q36m["kv_tokens"] / 1e6, 2),
            round(DATA["qwen38_no_mtp"]["kv_tokens"] / 1e6, 2),
            round(DATA["qwen38_mtp"]["depths"][3]["kv_tokens"] / 1e6, 2),
            round(DATA["gemma4_31b_qat"]["kv_tokens"] / 1e6, 2),
            round(DATA["qwen72_awq"]["kv_tokens"] / 1e6, 2),
        ],
        "million tokens in the GPU KV cache · * = bf16 KV, no FP8 on cc 8.0",
        highlight=0,
    )
    svg_grouped(
        OUT / "packing.svg",
        "How many agents fit in the KV pool",
        ["128K", "64K", "32K", "16K", "8K"],
        [
            ("27B MTP-3", _pack_vals(DATA["qwen38_mtp"]["mtp3"])),
            ("35B-A3B no MTP", _pack_vals(q36)),
        ],
        "concurrent agents = KV tokens / context (slot cap on the 27B recipe was 16)",
    )
    svg_grouped(
        OUT / "agents-compare.svg",
        "Aggregate tok/s vs parallel agents",
        ["1", "2", "4", "8", "16"],
        [
            ("27B MTP-3", _sweep_vals(DATA["qwen38_mtp"]["mtp3"])),
            ("35B-A3B no MTP", _sweep_vals(q36)),
            ("35B-A3B MTP-3", _sweep_vals(q36m)),
            ("Muse 30B", _muse_sweep(DATA["muse_glimmer_30b"])),
            ("Gemma4 31B", _muse_sweep(DATA["gemma4_31b_qat"])),
        ],
        "aggregate tok/s, short code prompt, max-num-seqs 16",
    )
    tp1, tp2, shard = DUAL["tp1"], DUAL["tp2"], DUAL["two_tp1_servers"]
    pp2 = DUAL["pipeline_parallel_pp2"]
    svg_grouped(
        OUT / "dual-card.svg",
        "Two 170HX, same 27B: tensor vs pipeline vs sharding",
        ["1 agent", "16 agents", "32 agents"],
        [
            ("TP=1 (one card)", [tp1["decode_tok_s"], tp1["agg_16_tok_s"], 0]),
            ("TP=2 (both cards)", [tp2["decode_tok_s"], tp2["agg_16_tok_s"], 0]),
            ("PP=2 (both cards)", [pp2["decode_tok_s"], pp2["agg_16_tok_s"], pp2["agg_32_tok_s"]]),
            ("2x TP=1 servers", [tp1["decode_tok_s"], shard["agg_16_tok_s"], shard["agg_32_tok_s"]]),
        ],
        "aggregate tok/s · no P2P (GNS), so TP=2 allreduce is host-staged at 5.85 GB/s",
    )
    muse = DUAL["muse_glimmer_pp2"]
    svg_grouped(
        OUT / "dual-card-muse.svg",
        "Muse Glimmer 30B: one card vs PP=2 across two",
        ["2 agents", "4 agents", "16 agents", "32 agents"],
        [
            ("One card", [113.22, 225.88, 789.78, 0]),
            ("PP=2 (both cards)", [muse["agg_2_tok_s"], muse["agg_4_tok_s"], muse["agg_16_tok_s"], muse["agg_32_tok_s"]]),
        ],
        "aggregate tok/s · one card wins until ~16 agents; the second card pays off at 32",
    )
    svg_bar(
        OUT / "dual-card-prefill.svg",
        "48K-token prefill: pipeline parallel wins outright",
        ["TP=1 (one card)", "TP=2 (both)", "PP=2 (both)"],
        [tp1["prefill_ttft_s"], tp2["prefill_ttft_s"], pp2["prefill_48k_ttft_s"]],
        "time to first token, seconds (lower is better)",
        highlight=2,
    )
    print("wrote", *sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
