# State (updated 2026-07-09)

## Current hypothesis
The text-to-image failure has been decomposed down to one hard sub-problem. Fixed/understood so far: the text path is healthy and sets the shared latent at the right scale; invertible strided-conv downsampling reconnected the generative pathway the one-way maxpools had severed; a top-down-authoritative generation schedule makes the caption drive the image per-pixel (PR 0 -> ~6). What REMAINS is amplitude: the top-down decode produces a dark, low-contrast, low-frequency image (gen brightness ~0.4x the true image, never sharp). A darkness diagnostic showed this is NOT the latent (text-set latent scale matches image-set) and NOT weight decay (decode weight-norm unchanged) -- it is the top-down PC decode producing the low-amplitude CONDITIONAL MEAN (the blurry average image consistent with the latent). The generative-training objective (two-phase: text-drive the latents, then bridge to the true image clamped at the bottom, local weight step) trains stably at a gentle schedule but does NOT sharpen -- it actually darkens over steps, because the clamped true image "helps" during the weight step so the decode never learns a strong standalone latent->image map. NEXT: redesign the generative objective to supervise the PURE top-down decode (no true-image-assisted bridge) toward the true image, i.e. teach sharp top-down generation. Separately, the broader banked question (does the coupling failure persist 156M -> 7.7B, judged at 8k against the bar) still stands via the capacity ladder.

## In flight
- The capacity ladder's Phase-0 probing (CAPACITY.md), exact sizes solved, memory and throughput measured per device, and the epoch-tier decision made (full 150 epochs at every rung, no reduction needed).
- Active drivers, `run_capacity_probe.py` (per-device probes, results in `capprobe_*.json`) and `run_coupling_capacity.py` (the ladder driver).

## Next steps
- Submit the capacity-ladder training runs per the Placement plan in CAPACITY.md (pc, E1L, and E1-Adam arms across 330M to 7.7B on the assigned GPUs).

## Open questions
- Which pre-registered branch the capacity ladder lands on. PC stays flat at chance through 7.7B while the baseline crosses at every size, PC crosses the bar at some size, or the baseline degrades at large capacity.

## How to update this file
Overwrite the fields above as work moves. Append the detailed run records to `docs/experiments/LOG.md` instead of here.
