# CAPACITY LADDER -- Phase 0 probe results and pre-submission projections (2026-07-05)

The paper's new experimental axis: does the coupling failure persist as the model grows from 156M to
7.7B parameters, at the decisive 8k-pair scale, latent retrieval primary, the banked bar (>3/2000)
unchanged. This file is the Phase-0 artifact the campaign requires before any training submission:
exact sizes, measured memory and throughput per device, projected wall-clocks, the epoch-tier decision,
and the baseline-feasibility calls. Probe scripts: run_capacity_probe.py (results in capprobe_*.json,
committed) and the ladder driver run_coupling_capacity.py (arm A reproduces the banked runner digit for
digit in smoke; auto-batch fallback; save/resume validated against an unbroken run).

## Sizes (exact, solved against the build shape arithmetic, V=49; anchor 1.5 -> 156,672,032 matches the
banked config to the digit)

| target | WMUL | actual params | off target |
|:--|--:|--:|--:|
| 330M | 2.18 | 330,164,875 | 0.0% |
| 700M | 3.18 | 700,795,531 | 0.1% |
| 1.5B | 4.66 | 1,500,584,411 | 0.0% |
| 3B | 6.59 | 3,002,452,749 | 0.1% |
| 7.7B | 10.555 | 7,700,274,219 | 0.0% |

## Probe table (fp32, 20 joint steps, largest fitting batch shown; NO FIT rows are recorded outcomes)

| size | device | pc (recipe step) | adam (E1 step) | lars (E1L step) |
|:--|:--|:--|:--|:--|
| 330M | L4    | B128, 5.6 GB, 0.616 s | B128, 11.8 GB, 0.572 s | B128, 6.9 GB, 0.460 s |
| 330M | MIG20 | B128, 5.8 GB, 0.841 s | B128, 11.9 GB, 0.587 s | B128, 7.0 GB, 0.515 s |
| 700M | L4    | B128, 9.3 GB, 1.091 s | B32, 19.3 GB, 0.576 s (tight) | B128, 12.3 GB, 0.875 s |
| 700M | MIG20 | B128, 10.8 GB, 1.424 s | NO FIT at any batch | B128, 12.3 GB, 0.986 s |
| 1.5B | L4    | B64, 13.7 GB, 1.212 s (B128 NO FIT) | NO FIT at B >= 16 | not probed (A100 numbers below) |
| 1.5B | A100  | B128, 16.1 GB, 0.629 s | B128, 44.8 GB, 0.401 s | B128, 22.8 GB, 0.329 s |
| 3B   | A100  | B128, 29.2 GB, 1.035 s | B16 only, 77.1 GB (too tight to trust; skipped) | B128, 41.9 GB, 0.704 s |
| 7.7B | A100  | B128, 72.6 GB, 2.267 s | NO FIT at any batch (m+v slots alone exceed the card) | crashed before probing (note below) |

7.7B notes. (1) pc at B128 ran 20 clean steps including LARS weight updates at 72.6 GB peak, 91 percent
of the card; the rung runs at B64 (one notch under the edge) with checkpointing on. (2) The probe
process later died with a TF C++ CHECK abort inside a reduction kernel while starting the lars config,
AFTER seven failed adam configs had fragmented the allocator; the pc evidence (20 clean steps in the
same process, earlier) says this is an allocator-state abort, not a size wall. The a1055 JSON was never
written (crash preceded the save); its pc/adam rows above are from the job log. The E1L 7.7B rung
doubles as the fresh-process check. (3) E1-Adam at 7.7B is infeasible at fp32 on this hardware, probed,
every batch: the baseline family for the ladder is therefore E1L (LARS-InfoNCE, stateless, fits every
rung, and it is the baseline with existing 156M anchors: held-out 7/4/5 per 2000 at 8k across seeds),
with E1-Adam run as corroboration at the sizes where it fits comfortably (330M, 700M, 1.5B).

## Projected wall-clock per PC arm (150 epochs, 8k pairs; steps = 150 x ceil(8000/B))

| size | device, batch | steps | projected wall |
|:--|:--|--:|--:|
| 330M | L4, B128 | 9,450 | 1.6 h |
| 700M | L4, B128 | 9,450 | 2.9 h |
| 1.5B | L4, B64 | 18,900 | 6.4 h |
| 3B | A100, B128 | 9,450 | 2.7 h |
| 7.7B | A100, B64 | 18,900 | 8-12 h (2.267 s at B128 bounds it; B64 s/step lands under that) |

EPOCH TIER DECISION: nothing approaches the 14-day flag, so every rung runs the full 150 epochs and the
156M anchor comparison needs no rerun. The epochs rule is satisfied with the original budget.

DEVIATIONS CARRIED (documented once here, recorded per rung in each JSON): reduced BATCHJ at 1.5B (64)
and 7.7B (64); the relaxation reduces with a sum over the batch so it is per-example and batch-invariant
by construction, the weight step is noisier at small batch and LARS normalizes its magnitude. E1L batch
drops to 128 on MIG at 700M and to 64 at 7.7B where memory requires it; batches are in the JSONs.

## Placement (submitted after this file; seed 0 everywhere; 3 seeds later at the largest completed size
plus any branch-flip size, per the pre-registration)

- L4 n7: pc 330M, then pc 1.5B chained afterany.
- L4 n8: pc 700M.
- MIG slices: E1L 330M, E1L 700M, E1-Adam 330M in parallel.
- A100, chained afterany in order: pc 3B, pc 7.7B (CKPT_EVERY=1000, READB=64), E1L 3B, E1L 7.7B,
  E1-Adam 1.5B, E1-Adam 700M. The chain keeps the card busy end to end.

## Pre-registered branches (unchanged from the campaign brief)

(a) PC flat at chance through 7.7B while the baseline crosses at every size: the capacity loophole is
closed and the claims extend from 156M to 7.7B. (b) PC crosses the bar at some size: major reframe;
seeds 1,2 plus a 20k rung at the crossing size before the word emerges is written. (c) The baseline
degrades at large capacity: reported as its own finding. Every rung also records movement (with the
norm/rotation decomposition run post-hoc on the saved checkpoint), align_cos, uniformity, diversity,
and i2i recon vs base; the capacity trend of mean-collapse alignment is a paper figure either way.
