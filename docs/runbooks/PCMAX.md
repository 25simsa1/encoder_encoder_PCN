# PCMAX — best-shot bidirectional PC (muPC + HEP error highways) at 156M/20k

Companion to `docs/runbooks/LADDER_V2.md` (the stability-recipe ladder this arm rides alongside)
and `docs/superpowers/specs/2026-07-24-pcmax-mupc-hep-design.md` (why and what). Driver:
`experiments/run_pcmax_capacity.py`. Arms: `PCMAX` (bidirectional interior states, HEP highways,
fully block-local weight updates, NO warmup — backprop-free by construction) and `Bmu` (the arm-B
recipe under muP parameterization only — the cheap ladder-hardening arm). Checkpoints
`cap_PCMAX_*` / `cap_Bmu_*` carry the 68 baseline keys + gains/td + an `__pcmax` marker;
`tools/mechanism_probe.py` and `tools/category_probe.py` handle both layouts (gated 2026-07-24).

## Gates (all PASSED on CPU, 2026-07-24 — do not resubmit code without re-running them)

1. PARITY: `RUNS1_SMOKE=1 RUNS1_ARMS=A,B PCMAX_PARITY=1 python3 experiments/run_pcmax_capacity.py`
   vs the same smoke on `run_coupling_capacity.py` — 31 numeric trace lines digit-identical AND the
   json arm payloads byte-equal. Proof the fork changes nothing.
2. SMOKE: `RUNS1_SMOKE=1 RUNS1_ARMS=PCMAX PCMAX_T_INFER=2 PCMAX_ALPHA=0.1 RUNS1_JOINTW=1.0` — F
   falls 2.0->0.43 over 24 steps, relax-F decreases inside every window, no NaN, ckpt reloads (68
   baseline keys + 26 additions), verdict prints. GD mode (`PCMAX_STATE_OPT=gd`): F strictly
   monotone over 8 steps.
3. PROBE: probe taps vs driver taps on the mup smoke ckpt max-abs-diff 5.4e-7 (<1e-5); old-vs-new
   probe code on a baseline ckpt bit-identical (0.0).
4. HIGHWAY LIVENESS: injection branch verified live (alpha=1e6 moves every interior state); at
   default scales the highway/local ratio is ~1e-5 — see calibration below. This is EXPECTED:
   HEP's boundary error is O(1), our InfoNCE latent gradient is not, so alpha is calibrated by the
   measured ratio, never copied from HEP's Table 4.

## Phase 1 (Colby, budget <=100 GPU-h, seed 0; submission on explicit go)

Recipe base (matches the ladder money cell): `RUNS1_JOINTW=1.0 RUNS1_LR=5e-3 RUNS1_EPOCHS=150
RUNS1_BATCHJ=64 RUNS1_WMUL=1.5 RUNS1_NTRAIN=20000 RUNS1_NEVAL=2000 RUNS1_COCO=train2017
RUNS1_DATA=/home/slsang29/coco_scale RUNS1_SEED=0`.

### Step 1 — alpha calibration (~1 GPU-h, L4/MIG)

One diag-only run: base env + `RUNS1_ARMS=PCMAX PCMAX_ALPHA=1.0 PCMAX_T_INFER=8 RUNS1_EPOCHS=1
RUNS1_CKPT=~/ladder/pcmax_cal`. Read the `[diag] step-1 mean ratio=R at alpha=1.0 -> alpha for
ratio 0.3 ~= A*` line (injection is linear in alpha, one measurement fixes the grid).

### Step 2 — alpha x T probe (~25-35 GPU-h, L4)

Short runs, base env + `RUNS1_EPOCHS=15 PCMAX_T_INFER=8`, alpha in {0, A*/3, A*, 3*A*} (alpha=0 is
HEP's own control), ckpt dirs `~/ladder/pcmax_a<k>`. Then the best alpha at `PCMAX_T_INFER=16`
(one run). Select on train-fit slope (lat_retr trajectory) + the diagnostic staying sane (nudges,
not overwrites: per-step RMS(dZ)/RMS(Z) at alpha must not dwarf the alpha=0 rows).

### Step 3 — full seed-0 runs (~45-55 GPU-h)

- PCMAX at best (alpha, T): base env + `PCMAX_FITSTOP=0.99 CAP_CKPT_EVERY=500
  RUNS1_CKPT=~/ladder/pcmax_w1.5_20k`. Walltime cap 40 h: a PCMAX step is ~4.5x (T=8) to ~9x
  (T=16) an arm-B step, so 150 ep uncapped would be ~58-113 h — FITSTOP is the budget lever, and
  an unfit-at-40h run is an instability/fit datum, not a failure to report. Resume works
  (single-arm PCMAX ckpt/resume extended and smoke-tested).
- Bmu (ladder-hardening control): base env + `RUNS1_ARMS=Bmu RUNS1_WARMUP=6000
  RUNS1_CKPT=~/ladder/bmu_w1.5_20k` (~13 h, arm-B cost).
- LR retry rule inherited from LADDER_V2: fit gate fails at 5e-3 -> ONE retry at 2e-3 -> else an
  instability-edge row.

### Step 4 — judgment

Driver battery prints the verdict; then the headline metric on the same split:

```
python3 tools/category_probe.py --bp ~/ladder/e1l_w1.5_20k/e1l_seed0.npz \
  --pc ~/ladder/pcmax_w1.5_20k/cap_PCMAX_w1.5_seed0.npz --ntrain 20000 --neval 2000 --seed 0 --coco train2017
# plus --pc <banked cs_B 20k ckpt> as the old-PC reference column
```

Pre-registered branches: category lift >>1.3x at matched fit => the dissociation was a
signal-delivery artifact, curable with local highways; ~1.3x at matched fit => the thesis is
hardened (best-shot scaled local PC still lacks binding); fails to fit => instability edge.
Seeds 1/2 (+50-80 GPU-h) only on an interesting seed-0. Ladder rungs 330M+ under muP LR transfer
are Phase 2, separate go.

## sbatch template (fork of ~/ladder/anchor.sbatch on Colby; swap driver + env)

```
#SBATCH --gres=gpu:l4:1 --time=40:00:00 ...   # per-step envs above
export RUNS1_ARMS=PCMAX PCMAX_ALPHA=<A> PCMAX_T_INFER=<T> PCMAX_FITSTOP=0.99 CAP_CKPT_EVERY=500
python3 experiments/run_pcmax_capacity.py
```

## Watch-items during runs

- The step print shows `relaxF a->b`: b<a must hold on trend; a growing a->b gap late in training
  means the highway term is dying with the InfoNCE gradient — the pre-registered fallback is
  RMS-normalized alpha (a new env knob, never a silent change).
- RMSNorm should make CAP_EAGER_WSTEP unnecessary at large WMUL — if the compiled weight step still
  NaNs at 6.59+, that is a finding for the stability appendix, and the knobs still exist.
