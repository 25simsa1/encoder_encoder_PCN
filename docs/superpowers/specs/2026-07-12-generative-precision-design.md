# Design: generative precision weighting, breaking the top-down contraction (the contraction swing)

Date: 2026-07-12

## Goal

Break the structural contraction of the top-down pathway so latent content can reach the pixels,
by adding per-layer PRECISIONS on the two existing prediction-error drive blocks in `update_state`
and running generation as a top-down-dominant inference schedule. Strictly bidirectional PC. First
test is READ-ONLY on the existing recon checkpoint, no retraining.

## Background (the pinned root)

The latent-autoencoder CHL retest pinned the root as structural. Under plain relaxation
(gamma 0) EVERY latent source decodes to exactly zero, and the only revival route (the gamma=1
replacement boost) is single-chain feedforward that destroys all but the deepest code's
information. The mechanism of the zero-collapse is the 1:1 drive balance in `update_state`, a free
state compromises between the top-down prediction from above and bottom-up consistency with the
(near-zero) states below, so the latent signal decays geometrically over the ~9-layer decode.
Note the PC energy's true minimum with only the top set IS the full ancestral top-down pass (zero
error on every top-down edge), the relaxation just cannot reach it against the bottom-up zero-drag.

## This is bidirectional PC (canonically so)

Precision-weighted prediction errors are core predictive-coding theory (Rao-Ballard, Friston, and
this project's own Stage-1.6 lesson that generative precisions must dominate). The update stays
the class's own local `update_state` on the same shared weights used both directions; the
precisions only reweight the two error blocks that already exist. No backprop, no separate
decoder, no optimizer.

## Component 1: per-layer error precisions

Add to `Conv2DPCNLayer` and `DensePCNLayer` two plain Python floats (trace-time constants),
`pi_td = 1.0` (weight on the next-layers drive block, the top-down consistency terms) and
`pi_bu = 1.0` (weight on the prev-layer drive block, the bottom-up consistency terms). In
`update_state` the two assign_subs become

  state -= state_lr * pi_td * (next-layers block)
  state -= state_lr * pi_bu * (prev-layer block)

Python-float multiplication by 1.0 is exact and folds at trace time, so the defaults are
byte-identical (the gate is the guard). `InputPCNLayer` has only the next-layers block and needs
no knob.

## Component 2: the generative inference schedule

Generation (and the latent-source probe's decode phase) sets `pi_bu` small (0.0 or a small eps)
on the image-path DECODE layers only, latents held fixed as always, text path and everything else
at the 1.0 defaults, and does NOT use the replacement boost. Each free decode state then adopts
the top-down prediction through the standard `update_state` averaging over ALL its next-layer
branches, so every latent contributes (unlike the single-chain boost) and nothing contracts to
zero. Restore 1.0 after.

## Component 3: the read-only decisive probe (no retraining)

Recon training already trained every per-edge top-down prediction (the d_pred term in
`update_wts`). The pure top-down cascade just chains those trained edges. So extend
`tools/latent_source_diag.py` with `--pi-bu` (decode phase uses the precision schedule instead of
the boost) and run it READ-ONLY on `ckpt_gen_best`, decode from image-set vs text-set latents plus
the per-scale swaps, at `pi_bu` 0.0 and 0.1 with the boost off.

Outcomes. (a) decode(img-set) shows the recognizable image, the decode already carries identity
and the remaining work is text-to-code alignment (fix B). (b) structured but wrong or weak, the
per-edge inverses need tuning, retrain briefly with the generative precisions in the CHL free
phase (well-posed now that the free phase cannot collapse to zero). (c) still template or zero,
report honestly, the shared-weight inverses do not chain, banking is the recommendation.

## Validation

- Local unit tests, with `pi_bu = 0` the state update is INVARIANT to the prev layer's content
  (and with `pi_td = 0` invariant to the next layer's), and defaults change nothing.
- The COCO64 gate vs the banked reference (layer files change, so re-gate; defaults are provably
  inert but the gate is the proof). NATIVE-143 stays deferred to a big GPU as before.
- The probe grids at pi_bu 0.0 and 0.1, judged visually plus mse/brightness/contrast, against the
  known template (boost) and zero (plain relax) baselines.

## Risks and unknowns

- The chained per-edge inverses may compose poorly (each was trained only against bottom-up-set
  states, and W W^T is not identity), giving distorted or low-fidelity output, that is outcome (b)
  and retraining with generative precisions is the follow-up.
- pi_bu = 0 removes all bottom-up moderation, states could over-shoot or ring, the 0.1 sweep point
  and the state clip are the guards.
- Touches `update_state` in the gate-critical layers, the byte-identical-default pattern (plain
  floats, exact multiply by 1.0) plus the gate covers it.

## Out of scope

Text-to-code alignment (fix B, only after the probe), retraining schedules, held-out, the ladder.
