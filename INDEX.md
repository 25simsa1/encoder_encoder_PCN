# Repo index

A map of what is where. The tree was reorganized into folders (Jul 2026): code, drivers,
docs, and notebooks now live in specific directories. Result **data** files (the `*.json`,
`grid_*.png`, and the tracked `*.log`) stay at the repo root on purpose — the drivers and the
numerical gate read and write them by run-directory-relative paths and construct result
filenames dynamically, so relocating them would break the gate and desync every run. Moving
the data cleanly needs an output-dir refactor of the drivers, deferred to after submission.

## Layout

```
(root)          core model + data + the frozen result/data files
experiments/    experiment drivers (run_*, dissociation, midscale*, port_*, analysis_*, ...)
tools/          analysis + probe tools that read checkpoints and write JSON
docs/           STATE.md, ARCHITECTURE.md, experiments/LOG.md, and:
  paper/          PAPER_DRAFT.md, PAPER_ADDENDUM_6.md
  runbooks/       RUN_*.md, CAPACITY.md, MIDSCALE.md, DISSOCIATION.md, E1_POD_RUNBOOK.sh
  notes/          literature + analysis notes (DEEP_LIT_PASS, LIT_PASS_2, REPO_ANALYSIS, ...)
  plans/          PORT_PLAN.md, PCN_FIX_PLAN.md
notebooks/      the .ipynb
results/        curated milestone grids + README (generation study)
tests/          unit tests
stage0/,1/      earlier staged MNIST/CIFAR pretraining
archive/        gitignored scratch (my decode grids, untracked run logs)
```

Run convention: scripts in `experiments/` and `tools/` import the core modules by name, so
run them from the repo root with the root on the path, e.g.
`PYTHONPATH=. python experiments/run_coupling_scale.py` (this is what the cluster jobs do).

## Newest work (generation study, Jul 2026 — start here if you're catching up)
Read `docs/STATE.md`, then the top of `docs/experiments/LOG.md`. Most-recently-touched:
`tools/text_nl_dense_ceiling.py`, `tools/text_nonlinear_ceiling.py`, `tools/nce_probe.py`,
`tools/text_ceiling.py`, `tools/text_align.py`, `tools/ridge_td.py`, `tools/relax_probe.py`,
`tools/gram_scales.py`, `train_coco64.py`, `conv_pcn_layer.py`, `dense_pcn_layer.py`,
`pcn_config.py` (the `COCO64_WIDE` config), and `results/`.

## Core model (the frozen recipe, at root)
- `encoder_encoder_pcn.py` -- the model
- `conv_pcn_layer.py`, `dense_pcn_layer.py`, `transformer_pcn_layer.py` -- layers
- `pcn_config.py` -- configs (NATIVE, COCO64_156M/GEN, COCO64_WIDE)
- `train_coco64.py`, `coco64_data.py`, `infonce.py`, `prep_coco.py` -- training + data

## Naming conventions
- `experiments/run_*.py` are drivers, `experiments/analysis_*.py` read saved checkpoints.
- Result files carry their config in the name: `res_<pairs>_<epochs>ep[_s<seed>|_T<inferdepth>].json`.
- `grid_*.png` / `*_grid.png` are sample panels mirroring a result file of the same stem.
- Every number in the paper traces to a root JSON (see `docs/paper/PAPER_DRAFT.md` Appendix A).

## Generation line (this study)
- `pcn_config.py` `COCO64_WIDE` (inter_dim 512) -- the config that broke the generation ceiling
- `train_coco64.py` -- `--config coco64_wide`, `--iso-scale`, `textdistill` mode, scale-robust
  InfoNCE coupling (`--infonce-lambda`)
- `tools/ridge_td.py` -- closed-form ridge solve for the untied top-down decode edges
  (`--ridge-conv` for the conv edges); the working decode recipe
- `tools/gram_scales.py`, `tools/relax_probe.py` -- localized the wide-net instability
- `tools/text_ceiling.py`, `text_nonlinear_ceiling.py`, `text_nl_dense_ceiling.py`,
  `nce_probe.py`, `text_align.py` -- the caption-to-image / coupling probes
- `results/` -- curated grids + README (image-to-image at mse 0.0402; caption side blocked
  by the coupling failure). Raw scratch grids are in `archive/decode_grids/`.

## Result data at root (paper-referenced, do not move without the driver output-dir refactor)
- COCO gate/held-out: `step1_coco_results.json`, `step1_heldout_results.json`
- coupling/scale: `coupling_scale_results_seed0.json`, `res_2k/8k/20k*.json`, `diag_jw0/1.json`
- factorial: `E1_results.json`, `E1L_results.json`, `BPonF*_results.json`, `coupling_unif_results.json`
- capacity: `capprobe_*.json`, `coupling_capacity_*.json`
- MNIST/dissociation: `dissoc_results*.json`, `midscale_*.json`, `runA/runB_results*.json`
- baselines: `golden_baseline.npz`, `golden_relaxed_baseline.npz`; figures: `grid_*.png`

## Prior lineage (in experiments/, predate the frozen recipe)
`port_*.py`, `repro_encoder_pcn.py`, `expdrv.py`, `durability300.py`, `sweep_relaxed.py`,
`run_instrumented.py`, `run_step2_coco_dissociation.py` -- verify against
`docs/paper/PAPER_DRAFT.md` Appendix A before removing any, since some paths may still be cited.
