# State (updated 2026-07-09)

## Current hypothesis
The text-to-image failure has now been decomposed. Invertible strided-conv downsampling reconnected the generative pathway (which the one-way maxpools had severed), and a top-down-authoritative generation schedule at inference makes the caption drive the image per-pixel and distinctly per caption (image PR 0 -> 6.93 as the top-down boost rises) -- so the structural block and the drive-balance are solved without retraining. What remains is that the decode produces artifacts/speckle rather than recognizable scenes, because it was trained both-clamped (reconstruction only) and never learned top-down self-sufficiency; the next lever is a PC-native generative-training objective. Separately, the broader banked question -- whether the coupling failure persists as the model scales 156M -> 7.7B, judged at the 8k-pair scale against the bar -- still stands via the capacity ladder.

## In flight
- The capacity ladder's Phase-0 probing (CAPACITY.md), exact sizes solved, memory and throughput measured per device, and the epoch-tier decision made (full 150 epochs at every rung, no reduction needed).
- Active drivers, `run_capacity_probe.py` (per-device probes, results in `capprobe_*.json`) and `run_coupling_capacity.py` (the ladder driver).

## Next steps
- Submit the capacity-ladder training runs per the Placement plan in CAPACITY.md (pc, E1L, and E1-Adam arms across 330M to 7.7B on the assigned GPUs).

## Open questions
- Which pre-registered branch the capacity ladder lands on. PC stays flat at chance through 7.7B while the baseline crosses at every size, PC crosses the bar at some size, or the baseline degrades at large capacity.

## How to update this file
Overwrite the fields above as work moves. Append the detailed run records to `docs/experiments/LOG.md` instead of here.
