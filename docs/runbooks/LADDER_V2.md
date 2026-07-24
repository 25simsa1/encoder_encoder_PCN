# LADDER V2 — capacity ladder under the stability recipe (category-transfer headline)

Companion to `docs/runbooks/CAPACITY.md` (Phase-0 sizes/probes/placement, all still valid) and
`docs/superpowers/specs/2026-07-23-ladder-plus-fa-campaign.md` (why). This file is the launch
recipe. Driver: `experiments/run_coupling_capacity.py`, arm B (warmup + joint at JOINTW), which
saves `cap_B_w{WMUL}_seed{SEED}.npz` in the cs_* key layout that `tools/category_probe.py` and
`tools/mechanism_probe.py` load unchanged (verified: arm-B smoke + npz reload, 2026-07-23). The
rung verdict now adjudicates from arm B and prints the matched-fit gate.

## The recipe (frozen; the money cell, job 9406)

`JOINTW=1.0 LR=5e-3 WARMUP=6000 EPOCHS=150 BATCHJ=64`. BATCHJ stays 64 at EVERY rung — the
relaxation is batch-invariant but the weight-step noise is not, and rungs must differ from the 156M
anchor in WIDTH ONLY. Cost of that choice: ~2x the CAPACITY.md step counts where B128 fit.

## Rung 0 — the anchor gate (blocks everything)

Reproduce the money cell through the CAPACITY driver (it is a scratch fork; this proves the fork
changes nothing at the reference point):

```
RUNS1_ARMS=B RUNS1_JOINTW=1.0 RUNS1_LR=5e-3 RUNS1_WARMUP=6000 RUNS1_EPOCHS=150 RUNS1_BATCHJ=64 \
RUNS1_WMUL=1.5 RUNS1_NTRAIN=20000 RUNS1_NEVAL=2000 RUNS1_COCO=train2017 \
RUNS1_DATA=/home/slsang29/coco_scale RUNS1_CKPT=~/ladder/w1.5_20k RUNS1_SEED=0 \
python3 experiments/run_coupling_capacity.py
```

ACCEPT: train lat_retr >= 0.99 (9406: 0.9973), held-out hits within noise of 3/2000, align/unif in
family with 9406. GPU reduction nondeterminism rules out bit-exactness; matching to printed
precision on the same card is the realistic bar. FAIL: stop, diff the fork against
run_coupling_scale.py before any larger rung.

## Rungs (after rung 0 passes)

WMUL per CAPACITY.md: 2.18 (330M), 3.18 (700M), 4.66 (1.5B), 6.59 (3B), 10.555 (7.7B). Devices per
its probe table (330M/700M/1.5B on L4, 3B/7.7B on A100; PC fits everywhere at B64, ckpting on for
7.7B). Two scales per rung, same env as rung 0 except:

- 20k primary: as rung 0 with `RUNS1_WMUL=<w> RUNS1_CKPT=~/ladder/w<w>_20k`.
- 8k banked bar: `RUNS1_NTRAIN=8000` (bar >3/2000 unchanged, invariant 3).

E1L baseline per rung (per-rung ckpt dir — e1l ckpt names lack wmul):

```
RUNS1_WMUL=<w> RUNS1_NTRAIN=20000 RUNS1_NEVAL=2000 RUNS1_COCO=train2017 \
RUNS1_DATA=/home/slsang29/coco_scale E1_SEEDS=0 E1_SAVE=1 E1_CKPT=~/ladder/e1l_w<w>_20k \
python3 experiments/run_E1_lars_infonce.py            # plus the NTRAIN=8000 variant
```

Category probe per rung (headline metric; DATA path inside the tool is the Colby cache):

```
python3 tools/category_probe.py --bp ~/ladder/e1l_w<w>_20k/e1l_seed0.npz \
  --pc ~/ladder/w<w>_20k/cap_B_w<w>_seed0.npz --ntrain 20000 --neval 2000 --seed 0 --coco train2017
```

## Gates and retries

- MATCHED-FIT GATE per rung: BOTH arms train lat_retr >= 0.95 (driver prints it for arm B; E1L logs
  its fit gate). PC fails at 5e-3 -> ONE retry at 2e-3; fails again -> record as an
  instability-edge row (paper section exists), rung excluded from the transfer comparison.
- Read at 700M before A100 submission: if either arm's gate fails there, stop and diagnose.
- Divergence or move < 40% -> the driver's own verdict handles it (stability datum / VOID).

## Wall-clock (B64, seed 0; 2x CAPACITY.md B128 projections, 20k = 2.5x 8k)

| rung | PC 8k | PC 20k | device |
|:--|--:|--:|:--|
| 330M | ~3.2 h | ~8 h | L4 |
| 700M | ~5.8 h | ~15 h | L4 |
| 1.5B | ~6.4 h | ~16 h | L4 (B64 native) |
| 3B | ~5.4 h | ~14 h | A100 |
| 7.7B | ~8-12 h | ~20-30 h | A100 (B64) |

E1L adds ~1-5 h per rung per scale. KNOWN GAP: checkpoint/resume is arm-A-only, so the 7.7B/20k
arm-B rung (~20-30 h) runs unresumable. Submit it with a generous walltime after the 8k rung
banks; if the queue's limit makes that impossible, extending CAP_RESUME to arm B (second RNG
stream in the saved state) is the follow-up task to do FIRST.

## Order

1. Rung 0 anchor (20k). 2. 330M + 700M both scales + E1L (L4s/MIGs in parallel). 3. READ. 4. 1.5B,
then A100 chain 3B -> 7.7B (8k then 20k), E1L rungs interleaved. 5. Replication (3 seeds) at the
largest completed size + any branch-flip size, 20k only. 6. FA (separate runbook line once
`run_FA_lars_infonce.py` lands): 156M/20k x 3 seeds, then category probe with `--bp` pointing at
the FA ckpt.
