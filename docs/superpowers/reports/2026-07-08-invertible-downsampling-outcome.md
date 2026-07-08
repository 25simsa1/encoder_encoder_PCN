# Invertible downsampling — outcome report

Date: 2026-07-08
Plan: `docs/superpowers/plans/2026-07-07-invertible-downsampling.md`
Spec: `docs/superpowers/specs/2026-07-07-invertible-downsampling-design.md`

## What shipped (code, all reviewed clean, on master, NATIVE gate held twice)

Stride-2 shared-weight conv downsampling as an opt-in replacement for the image
maxpools, so top-down generative drive can flow through the downsampling:
- `conv_pcn_layer.py` — `Conv2DPCNLayer` gains `stride` (default 1); threaded through the
  forward conv, `predict_prev` (transpose, restores the recorded input size), `pred_loss_d_input`,
  `update_state`, `update_wts`. `stride=1` byte-identical (commit 7e43365).
- `pcn_config.py` — `downsample` field (default `'maxpool'`) + `COCO64_GEN` (commit 1cd4693).
- `encoder_encoder_pcn.py` — `_build_downsample` branch: maxpool or stride-2 channel-preserving
  conv at the 4 sites (commit a9c822c).
- `train_coco64.py` — `--config` selector; conv-activation override on stride-1 convs only, so
  the downsamplers stay linear (commit 410b785).

Gates: NATIVE `GATE_MATCH nlayers=143` at Tasks 1 and 3 (conv change + config branch both inert
on NATIVE); COCO64_GEN builds with 5 aliases, both generation directions finite. Final
whole-branch review clean (cross-commit consistency, `input_shape` set before any strided
`predict_prev`, boundary shapes, `Conv2DBackpropFilter [1,2,2,1]`).

## Experiment: retrain COCO64_GEN 2k + text→image retest (JOB 8436, H200)

Config: `--config coco64_gen`, pairs 2000, lr 1e-3, wd 3e-2, state_clip 400, gelu conv (the 4
strided downsamplers stayed linear), relax 15, batch 8, 15 epochs, compiled relaxed schedule
(~0.39 s/step). Trained STABLY: energy descended to the fit floor (0.0086 → 0.0040 → 0.0060 over
ep4/8/13), max|state| ~110–122 (well under 400), no divergence, `TRAIN_DONE`. So invertible
downsampling reconstructs as well as maxpool (comparable to the ~0.0038 gelu-maxpool best) and
does not destabilize training.

Text→image retest (`tools/gen_retest.py`, 150 relax, in-sample pairs, model's own `test_step`):
- **text→image (zero image init, caption clamped): still a uniform blob** — `image_PR=0.000`,
  every PNG 91 bytes / std 0.00000 (exactly zero output). No caption-varying structure.
- **reconstruction (real image init): intact** — `PR=3.033`, structured 5.5–8.9 KB PNGs (verified
  visually: the true scene reconstructed).
- image→caption: the probe's caption readout collapsed to a constant (`666…`) — the same
  readout/init artifact seen on prior checkpoints, not informative.

## Verdict

**Invertible downsampling is NECESSARY but NOT SUFFICIENT.** It removes the structural block (the
one-way maxpool that had no top-down inverse) and reconstructs cleanly, but text→image still
produces a uniform blob under the standard symmetric relaxation. Two obstacles remain, both
flagged by the earlier diagnostics and explicitly out of scope for this plan (which scoped only
the structural invertibility):

1. **Drive-balance persists.** With the image unclamped from a zero init, bottom-up drive from the
   zero image still dominates the (now-connected) top-down drive during the symmetric relaxation,
   so the text-set latent can't move the image off zero. (The earlier `topdown_drive_diag`/`topdown_gen_test`
   showed inference needs an explicit top-down boost to un-collapse; the standard relaxation does not
   supply it, even with the pathway reconnected.)
2. **The decode was never trained to be top-down-self-sufficient.** COCO64_GEN was trained the same
   both-clamped way (image always pinned → reconstruction), so the weights never had to generate an
   image from text alone.

## Next options (localized by this result)

- **Cheap probe:** run the full top-down-boosted generation on COCO64_GEN (now the boost propagates
  through the invertible downsamplers — no maxpool skip). If it yields structure, text→image works
  with a top-down-authoritative generation SCHEDULE at inference (no retraining needed beyond
  invertibility); if not, the weights themselves need generative training.
- **Generative training objective (the deferred half):** a staged relaxation / image-unclamped
  training so the shared weights learn top-down self-sufficiency.
- **Bank:** invertible downsampling shipped + gated; text→image remains blocked by drive-balance +
  non-generative training — a clean incremental result for the writeup.

## Artifacts

Checkpoint (cluster): `ckpt_gen_best` (ckpt-18). Log `gen_8436.log`. Retest images (local):
`gen_retest_out/` — `t2i_*` (all black), `recon_*`/`true_*` (match). Throwaway `tools/gen_retest.py`,
`tools/run_gen.sh` are runners, not committed.
