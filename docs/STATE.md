# State (updated 2026-07-09)

## Current hypothesis
The cross-modal coupling failure (an image-dominated shared latent that blocks text-to-image generation) has resisted every intervention tried so far, including a PC-native InfoNCE coupling and invertible strided-conv downsampling. The open experimental question is whether it persists as the model scales from 156M to 7.7B parameters, judged at the 8k-pair scale against the banked bar.

## In flight
- The capacity ladder's Phase-0 probing (CAPACITY.md), exact sizes solved, memory and throughput measured per device, and the epoch-tier decision made (full 150 epochs at every rung, no reduction needed).

## Next steps
- Submit the capacity-ladder training runs per the Placement plan in CAPACITY.md (pc, E1L, and E1-Adam arms across 330M to 7.7B on the assigned GPUs).

## Open questions
- Which pre-registered branch the capacity ladder lands on. PC stays flat at chance through 7.7B while the baseline crosses at every size, PC crosses the bar at some size, or the baseline degrades at large capacity.

## How to update this file
Overwrite the fields above as work moves. Append the detailed run records to `docs/experiments/LOG.md` instead of here.
