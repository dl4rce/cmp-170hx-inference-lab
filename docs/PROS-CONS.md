# Pros and cons of a CMP 170HX for local LLM inference

Measured 2026-08-21 on one unlocked card (64 GB, FMA on, PCIe Gen2 ×16, vLLM 0.27.1). Not a product review of every board on eBay.

## Pros

- **64 GB HBM2e** after the 8 GB SKU unlock. A used 3090 is 24 GB; a 5090 is 32 GB. This is the only reason to care.
- **HBM bandwidth** ~1.26 TB/s in our microbench. Decode loves that more than GDDR on 16 GB workstation cards.
- **Qwen3.6 35B-A3B AWQ** (~3B active): **135 tok/s** single stream, **1 261 tok/s** at 16 agents, **3.16M** FP8 KV (24× at 128K). MTP-3 does not help single-stream on this MoE.
- **Qwen3.8 27B W4A16** at 128K with a ~1.0–1.2M token FP8 KV pool. MTP-3: **88 tok/s** single stream, **641 tok/s** aggregate at 16 slots.
- **Native 256K** on 27B: 1.10M KV, exact retrieval at 200k prompt tokens.
- **72B W4A16 actually loads** (38.8 GiB in ~20 s on Gen2 ×16). A 24 GB 3090 cannot hold that checkpoint. Native context is **32K**, decode **26 tok/s**, already at the 250 W cap. Standard vLLM random ISL/OSL: 25→10 tok/s as prompts go 128→8k; 4-wide **79 tok/s** aggregate.
- Single-GPU, no tensor-parallel PCIe tax. Dual 16 GB cards doing TP=2 over PCIe without NVLink lose a lot of the 27B decode (31 tok/s vs 57 tok/s here without MTP).

## Cons

- **Not an A100.** Same family (GA100), not the same product: no NVLink, no native FP8/FP4 (cc 8.0), no display, mining VBIOS/OTP, driver 610.x unlock path.
- **PCIe is Gen2**, often **×4** unless someone soldered the missing AC-coupling capacitors. ×16 only helped **load / H2D / vision**, not decode. Stock ×4 is ~1.7 GB/s H2D vs 6.6 GB/s on ×16.
- **250 W** and mining cooling. 16-agent 27B already 248–253 W. 72B hits TDP on **one** stream. A quiet desktop without a proper cooler is a bad idea.
- **Unlock + SKU lottery.** 8 GB→64 GB vs 10 GB→40 GB. 80 GB claims are unstable. Software unlock is experimental; Secure Boot off; persistence is patched modules.
- **MTP-4 is not free speed.** Depth 3 peaked; depth 4 was slower and smaller KV.
- **Tokens per watt** at 1–4 agents on **dense 27B** does not beat a 2×70 W PRO 2000 box. The 170HX wins when you fill 8–16 streams, when the model will not fit in 16–24 GB, or on a **~3B-active MoE** (Qwen3.6) that actually uses the 64 GB KV pool.
- **Multi-card 300B MoE** over this link, with no NVLink and no FP8, is a science project. Public four-card write-ups had to avoid tensor parallel.

## What 1 500–1 800 € is paying for

That band is “A100-adjacent” marketing after the unlock went public (cards that were ~$100–200 jumped past $1 000; see Wccftech). You still have to add:

1. Time and risk of the **software unlock** (or a seller who already did it — trust, but verify 64 536 MiB and a compute microbench).
2. Optional **solder mod** if you care about load time and vision H2D.
3. A **real cooler** and a PSU/case that can take 250 W of miner noise and heat.

For 27B production, a boring 70 W dual-16GB workstation is still the sane appliance. For a lab that needs 64 GB in one card, the 170HX is interesting **well below** A100 money. 1 500–1 800 € plus unlock plus hardware plus cooling is buying the story, not the silicon we measured.
