# Campaign plan — capacity ladder + a second local rule (novelty hardening)

**Why.** The current result (matched-fit category-transfer dissociation at 156M/20k) is one scale,
one PC instantiation. Reviewers file that under "local rules underperform BP, known since Bartunov
2018". Two additions convert it into a claim reviewers cannot dismiss: (1) the CAPACITY LADDER shows
whether the dissociation persists 156M -> 7.7B (the project's registered thesis, still unrun), and
(2) a FEEDBACK-ALIGNMENT arm shows whether the identity-not-value failure is a property of LOCALITY
or of this PC recipe. Mock of the target figure: `2026-07-23-ladder-mock.png` (same directory).

## What changed since the Phase-0 artifact (docs/runbooks/CAPACITY.md, 2026-07-05)

CAPACITY.md's sizes (WMUL 2.18/3.18/4.66/6.59/10.555 -> 330M..7.7B), memory probes, device
placement, and epoch-tier decision all still hold. Three things are superseded:

1. **PC arm recipe.** The ladder must run the STABILITY RECIPE that produced the money cell:
   `RUNS1_JOINTW=1.0 RUNS1_LR=5e-3 RUNS1_WARMUP=6000 RUNS1_EPOCHS=150` (jointw=0 washes out train
   coupling; lr 2e-2 diverges late at 20k). Everything else per `experiments/run_coupling_scale.py`.
2. **Headline metric.** Instance top-1 hits stay as the banked 8k bar (invariant 3, unmoved), but
   the paper's metric is now CATEGORY precision@10 lift (`tools/category_probe.py`), which needs
   saved checkpoints at every rung (PC `RUNS1_CKPT`, E1L `E1_SAVE=1`).
3. **Matched-fit gate.** A rung only counts if BOTH arms fit train coupling (train lat_retr >= 0.95).
   A PC rung that cannot fit at 5e-3 gets one lr retry (2e-3); a second failure is recorded as an
   instability-edge data point (paper already has that section), not silently dropped.

## Phase A — port + smoke (blocks everything; ~1 day, near-zero GPU)

- A1. Thread the stability recipe + ckpt saving through the ladder driver at each WMUL; verify the
  156M anchor reproduces the banked money cell digit-for-digit from a saved ckpt (the gate that the
  ladder code changes nothing).
- A2. Per-rung 2-epoch smoke at target batch on the target device (memory + energy sanity) before
  any full submission. CAPACITY.md probes say fp32 fits everywhere PC runs (7.7B at B64, ckpting on).

## Phase B — the ladder (the paper's spine)

Rungs: 330M, 700M, 1.5B, 3B, 7.7B (156M anchors exist). Arms: PC (stability recipe) and E1L
(`experiments/run_E1_lars_infonce.py`, the baseline that fits every rung; E1-Adam corroboration only
at 330M/700M/1.5B where it fits). Two data scales:

- **20k pairs** (train2017 cache, NEVAL 2000) — PRIMARY, where the category headline lives and BP
  demonstrably works (11-13 instance hits, ~2.5x lift).
- **8k pairs** — the pre-registered banked bar (>3/2000, latent retrieval primary). Invariant 3 says
  this bar decides; it runs unchanged.

Per rung record: train lat_retr (fit gate), held-out instance hits, category prec@10 lift per
category, align_cos/uniformity/diversity, movement + norm/rotation decomposition, i2i recon. Seed 0
everywhere first; 3 seeds at the largest completed size plus any branch-flip size (per the
pre-registration).

**Wall-clock (seed 0, from CAPACITY.md projections; 20k ~= 2.5x the 8k number):**

| rung | PC 8k | PC 20k | E1L (both scales) |
|:--|--:|--:|--:|
| 330M (L4) | 1.6 h | ~4 h | ~1 + 3 h |
| 700M (L4) | 2.9 h | ~7 h | ~2 + 5 h |
| 1.5B (L4/A100) | 6.4 h | ~16 h | ~2 + 5 h |
| 3B (A100) | 2.7 h | ~7 h | ~2 + 5 h |
| 7.7B (A100, B64) | 8-12 h | ~20-30 h | ~5 + 12 h |

Seed-0 total ~105-130 GPU-hours spread over L4s/MIGs/A100 per CAPACITY.md placement; replication
wave (3 seeds at 7.7B + one flip size, 20k only) adds ~60-90 h. Chained afterany, the A100 line is
~1 week of card time; nothing approaches the 14-day flag.

**Pre-registered branches (category-lift version, extending CAPACITY.md's):**
(a) BP lift stays ~2-2.5x while PC stays ~1-1.3x through 7.7B -> the limitation is capacity-robust,
claims extend, strongest paper. (b) PC lift rises toward BP at some size -> major (positive!)
reframe: "local PC needs Nx the capacity for the same cross-modal abstraction"; 3 seeds + a 20k rung
at the crossing before any writing. (c) BP degrades at scale -> its own finding. All three are
publishable; the ladder cannot null.

## Phase C — feedback alignment (the locality control)

- C1. Fork `run_E1_lars_infonce.py` -> `run_FA_lars_infonce.py`: identical forward/objective/LARS;
  backward replaces each weight transpose with a FIXED random matrix (Lillicrap 2016), implemented
  via `@tf.custom_gradient` on the dense/conv matmuls. No other change.
- C2. Validation gates: (i) with feedback = true transposes the fork reproduces E1L's first 50 steps
  digit-for-digit; (ii) random feedback trains and passes the same train-fit gate.
- C3. Runs: 156M/20k x 3 seeds, same splits, ckpts, category probe. This is cheap (~E1L cost, <10 h
  total). Ladder extension for FA only if the 156M result is interesting.

Readout: FA fits train but shows PC-like ~1.3x flat lift -> "local credit assignment fails to build
transferable cross-modal categories" (a rule-family claim, much stronger). FA shows BP-like ~2.5x ->
the failure is PC-specific (identity-vs-value mechanism becomes the star). Either answer sharpens
the paper.

## Order and decision points

1. Phase A now (no GPU contention).
2. Phase C1/C2 in parallel with Phase A (pure implementation).
3. Phase B seed-0 wave; read at 700M — if either arm's fit gate fails there, stop and fix before
   burning A100 time. Full read after 7.7B decides branch (a)/(b)/(c) and the replication wave.
4. FA runs any time after C2 (fits on small slices).

## Figure this buys (see mock)

Panel (a): category lift vs params (log-x), BP and PC curves with seed dots, base-rate line — the
three branches drawn as the pre-registered outcomes. Panel (b): rule-family bars at 156M/20k — BP
vs PC vs FA against base rate. Together they answer "does it scale away?" and "is it locality?",
the two questions the current single-scale figure cannot.
