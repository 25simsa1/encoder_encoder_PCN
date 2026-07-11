# State (updated 2026-07-11)

## Current hypothesis
The text-to-image failure has been decomposed down to one hard sub-problem. Fixed/understood so far: the text path is healthy and sets the shared latent at the right scale; invertible strided-conv downsampling reconnected the generative pathway the one-way maxpools had severed; a top-down-authoritative generation schedule makes the caption drive the image per-pixel (PR 0 -> ~6). What REMAINS is amplitude: the top-down decode produces a dark, low-contrast, low-frequency image (gen brightness ~0.4x the true image, never sharp). A darkness diagnostic showed this is NOT the latent (text-set latent scale matches image-set) and NOT weight decay (decode weight-norm unchanged) -- it is the top-down PC decode producing the low-amplitude CONDITIONAL MEAN (the blurry average image consistent with the latent). The generative-training objective (two-phase: text-drive the latents, then bridge to the true image clamped at the bottom, local weight step) trains stably at a gentle schedule but does NOT sharpen -- it actually darkens over steps, because the clamped true image "helps" during the weight step so the decode never learns a strong standalone latent->image map. The generative objective was then redesigned as a contrastive-Hebbian (CHL) pass that supervises the PURE top-down decode toward the true image, but it (and every prior generative objective) DESTABILIZES at ~ep13 the same way: NORM INFLATION (the beta-less LARS trust ratio grows with ||w||, a positive feedback that inflates weights and states until the state clip and the energy explode). So the objective is not the blocker; the norm-inflation instability is, and it caps how long any generative training can run.

CURRENT FIX (in flight): a PC-native weight-normalization stabilizer. Each opt-in conv/dense weight is reparameterized w = g_mag * v/||v|| (per-output-unit magnitude + unit direction, the SAME shared weight both directions so it stays bidirectional PC), and update_wts splits the class's own local gradient into a radial magnitude step + a tangential direction step so ||w|| cannot run away; LARS trust is dropped for these layers. Opt-in via --weight-norm, byte-identical when off. DONE so far: both layers (conv c0a4d42, dense e0ba22e, 7 unit tests), and the COCO64 inertness gate PASSES (GATE_MATCH nlayers=88, current flag-off code byte-identical to the pre-change reference; the canonical NATIVE-143 gate is deferred to a big-GPU window since H200 is drained). PENDING: wire --weight-norm into train_coco64 with g_mag persistence, then the make-or-break CHL retrain (does training now hold past ep13 without the state pinning at the clip) + darkness/text-to-image retest. Fallback if it does not fully stabilize: fix g_mag at init (learn direction only).

Separately, the broader banked question (does the coupling failure persist 156M -> 7.7B, judged at 8k against the bar) still stands via the capacity ladder.

## In flight
- The capacity ladder's Phase-0 probing (CAPACITY.md), exact sizes solved, memory and throughput measured per device, and the epoch-tier decision made (full 150 epochs at every rung, no reduction needed).
- Active drivers, `run_capacity_probe.py` (per-device probes, results in `capprobe_*.json`) and `run_coupling_capacity.py` (the ladder driver).

## Next steps
- Submit the capacity-ladder training runs per the Placement plan in CAPACITY.md (pc, E1L, and E1-Adam arms across 330M to 7.7B on the assigned GPUs).

## Open questions
- Which pre-registered branch the capacity ladder lands on. PC stays flat at chance through 7.7B while the baseline crosses at every size, PC crosses the bar at some size, or the baseline degrades at large capacity.

## How to update this file
Overwrite the fields above as work moves. Append the detailed run records to `docs/experiments/LOG.md` instead of here.
