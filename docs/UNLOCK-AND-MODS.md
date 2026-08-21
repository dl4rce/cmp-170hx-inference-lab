# Unlock, SKU lottery, and the hardware mod

This lab did **not** perform the unlock. The rented card already reported 64 GB, unlocked FMA, and PCIe **Gen2 ×16**. The notes below are from public community sources so a buyer can price the *work*, not just the board.

## Two different problems

| What you want | How | Typical stock | After |
|---|---|---|---|
| Compute + VRAM geometry | Software unlock (patched `nvidia-open` 610.43.x) | 8 GB or 10 GB visible, FMA throttled, often Gen1 | 64 GB (8 GB SKU) or 40 GB (10 GB SKU), full SM throughput, **Gen2 speed** |
| PCIe **width** ×16 | Hardware: populate missing AC-coupling capacitors | Often **×4** (12 of 16 lanes unpopulated) | ×16. Independent of the Gen2 speed unlock |

Gen2 vs ×16 are orthogonal. Software gives you Gen2. Width is solder. Neither one is PCIe Gen3/Gen4, NVLink, ECC, or native FP8/FP4.

Primary public write-ups:

- Unlock tool: [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)
- Technical wiki (silicon, OTP, cooling, open problems): [Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx)
- Price spike after the unlock landed: [Wccftech, 2026](https://wccftech.com/nvidia-cmp-170hx-8-10-gb-prices-explode-over-1000-usd-as-tool-unlocks-hidden-64-80gb-vram/)

This repository does **not** ship patches, ROP chains, or register recipes. Use the projects above, and budget a cold power-cycle plus Secure Boot off.

## SKU lottery

- **8 GB / device id `10de:20c2` (often Hynix):** community consensus is this is the 64 GB unlock.
- **10 GB / `10de:2082` (often Samsung):** 40 GB is the stable story. 80 GB geometry has been tried and reported unstable (GSP/DMA faults). Do not buy “80 GB 170HX” marketing.

FMA lock is also a lottery until the unlock is applied and `nvidia-smi` / a microbench show real FP32/BF16 throughput.

## Cooling and the rest of the board

These were mining cards. Expect a blower or a barely adequate shroud, **250 W**, no display output, and usually no NVENC. Putting one in a quiet desktop without a real cooler is the part people skip in 1 500–1 800 € listings.

NVLink fingers may be present; the interface ICs are typically unpopulated. Multi-GPU tensor parallel over this PCIe is a trap (see also public DeepSeek-V4-on-four-170HX write-ups that had to use **pipeline** parallel, not TP).

## Reddit (field reports, not vendor copy)

- Unlock announcement / skepticism thread: [r/LocalLLaMA — Nvidia CMP 170HX Unlock 8GB to 64GB](https://www.reddit.com/r/LocalLLaMA/comments/1v0a7wn/)
- Later field-report thread: [r/LocalLLaMA — Any CMP 170HX field report](https://www.reddit.com/r/LocalLLaMA/comments/1vbf1s0/)

Hype in those threads still outruns the card: “5090 speed with 64 GB” is not what we measured. 64 GB is real. 2025–26 inference formats (FP8/FP4) are not.
