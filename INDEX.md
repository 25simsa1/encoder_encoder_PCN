# Repo index

A map of what is where. Tracked files are NOT moved into folders on purpose: the paper draft,
the RUN logs, and the Slurm job scripts all reference result and script paths inline, and jobs
are still writing here, so a physical reorg waits until after submission. This file is the
navigation layer instead. Pure scratch (my session decode grids, untracked run logs) lives in
`archive/` (gitignored) to keep the working dir readable.

## Newest work (generation study, Jul 2026 — start here if you're catching up)
Most-recently-touched files, newest first: `tools/text_nl_dense_ceiling.py`,
`docs/experiments/LOG.md`, `tools/text_nonlinear_ceiling.py`, `docs/STATE.md`,
`tools/nce_probe.py`, `train_coco64.py`, `tools/text_ceiling.py`, `tools/text_align.py`,
`tools/ridge_td.py`, `tools/relax_probe.py`, `tools/gram_scales.py`, `conv_pcn_layer.py`,
`dense_pcn_layer.py`, `pcn_config.py` (the `COCO64_WIDE` config), and `results/`.
Read `docs/STATE.md` then the top of `docs/experiments/LOG.md` for the current status.

## Generation line (this study)
- `pcn_config.py` -- `COCO64_WIDE` (inter_dim 512), the config that broke the generation ceiling
- `train_coco64.py` -- adds `--config coco64_wide`, `--iso-scale`, `textdistill` mode, and the
  scale-robust InfoNCE coupling fix (`--infonce-lambda`)
- `tools/ridge_td.py` -- closed-form ridge solve for the untied top-down decode edges
  (`--ridge-conv` solves the conv edges too); the working decode recipe
- `tools/gram_scales.py`, `tools/relax_probe.py` -- localized the wide-net instability
- `tools/text_ceiling.py`, `tools/text_nonlinear_ceiling.py`, `tools/text_nl_dense_ceiling.py`,
  `tools/nce_probe.py`, `tools/text_align.py` -- the caption-to-image / coupling probes
- `results/` -- curated milestone grids + README (image-to-image at mse 0.0402; caption side
  blocked by the coupling failure). Raw scratch grids are in `archive/decode_grids/`.

Naming conventions worth knowing:
- `run_*.py` are experiment drivers, `analysis_*.py` are post-hoc tools that read saved checkpoints.
- Result files carry their config in the name: `res_<pairs>_<epochs>ep[_s<seed>|_T<inferdepth>].json`.
- `grid_*.png` and `*_grid.png` are the sample panels that mirror a result file of the same stem.
- Every number cited in the paper traces to one of these JSONs (see `PAPER_DRAFT.md` Appendix A).

## Core model (the frozen recipe)
- `encoder_encoder_pcn.py` -- the model
- `conv_pcn_layer.py`, `dense_pcn_layer.py`, `transformer_pcn_layer.py` -- layers

## Primary COCO drivers (recipe-frozen lineage)
- `run_step1_coco_gate.py` -> `step1_coco_results.json` -- in-sample gate (400 pairs)
- `run_step1_coco_heldout.py` -> `step1_heldout_results.json` -- the held-out test
- `run_infonce_warmup_coco.py` -- InfoNCE warm-up vs baseline + A_long control
- `run_coupling_scale.py` -> `coupling_scale_results_seed0.json`, `res_2k/8k/20k*.json` -- data-scale
  and matched-epochs curve. RUNS1_* env surface; N_INFER and RUNS1_NINFER for the T-sweep.

## Two-factor factorial (controls and baselines)
- `run_E1_bp_clip_baseline.py` -> `E1_results.json` -- from-scratch BP CLIP (Adam+InfoNCE), achievability
- `run_E1_lars_infonce.py` -> `E1L_results.json` -- LARS+InfoNCE, the optimizer control
- `run_BPonF.py` -> `BPonF_results.json` -- backprop on the energy F, the objective control
- `run_BPonF_freelatent.py` -> `BPonF_freelatent_results.json` -- free-latent variant (unrolled relax)
- `run_coupling_unif.py` -> `coupling_unif_results.json`, `unif_unif_*.json` -- the repair test (F_unif)
- `diag_jw0.json`, `diag_jw1.json` -- warm-up washout diagnostic

## Capacity ladder (in progress)
- `run_capacity_probe.py`, `run_capacity_nanprobe.py` -> `capprobe_*.json` -- memory/throughput probe
- `run_coupling_capacity.py` -> `coupling_capacity_*.json` -- the 330M -> 7.7B rungs
- log: `CAPACITY.md`

## MNIST scaling and dissociation (Story A, in-sample)
- `dissociation.py` -> `dissoc_results*.json` -- the 2x2x2 matrix, 3 seeds (DISSOC_SEED env)
- `midscale.py`, `midscale_actgrid.py`, `midscale_seeds.py`, `midscale_staged.py`
  -> `midscale_*.json`, `actgrid_results.json`, `seeds_results.json` -- the positive control
- `run_A_scale_mup.py` -> `runA_results_mup.json` -- muP scaling
- `run_B_scale_push.py` -> `runB_results.json` -- scaling to ~3B

## Analysis tools (read checkpoints, write JSON)
- `analysis_move_decomp.py` -> `movedecomp/` -- weight movement norm/rotation decomposition
- `analysis_latent_geometry.py` -> `analysis_latent_geometry_results.json` (+ `.md`) -- E4 geometry battery
- `analysis_pooled_stats.py` -> `pooled_stats.json` -- exact binomial pooled stats
- `analysis_seed1_forensics.py` -> `seed1_forensics.json` -- the 8k seed-1 anomaly

## Documentation
- `PAPER_DRAFT.md` -- the paper. `PAPER_ADDENDUM_6.md` -- writing-stack addendum.
- `README.md` -- project readme. This file (`INDEX.md`) -- the map.
- Literature: `LIT_PASS_2.md`, `DEEP_LIT_PASS.md` (`LIT_REVIEW.md` is local-only, gitignored).
- Run logs: `RUN_STEP1.md`, `RUN_STEP1_HELDOUT.md`, `RUN_UNIF.md`, `RUN_A.md`, `RUN_B.md`,
  `CAPACITY.md`, `MIDSCALE.md`, `DISSOCIATION.md`.
- Planning: `REPO_ANALYSIS.md`, `PORT_PLAN.md`, `PCN_FIX_PLAN.md`.

## Data and infra
- `prep_coco.py` -- COCO cache builder (val2017 / train2017)
- `tests/` -- unit tests
- `stage0/`, `stage1/` -- earlier staged pretraining experiments (MNIST/CIFAR)

## Earlier iterations (predate the frozen recipe, kept for provenance)
`port_7b.py`, `port_7c.py`, `port_redesign.py`, `port_redesign2.py`, `port_smoke.py`,
`repro_encoder_pcn.py`, `expdrv.py`, `durability300.py`, `sweep_relaxed.py`,
`run_instrumented.py`, `run_step2_coco_dissociation.py`. Not part of the current result set;
verify against `PAPER_DRAFT.md` Appendix A before removing any, since some paths may still be cited.
