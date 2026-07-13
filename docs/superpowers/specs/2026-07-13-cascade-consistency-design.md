# Design: cascade-consistency training, per-edge calibration of the top-down cascade

Date: 2026-07-13

## Goal

Make the top-down cascade render latent identity by calibrating EACH EDGE against a bounded,
per-layer ground truth, the recon states, instead of the compounded end-to-end error that
destabilized the calibration CHL. Strictly bidirectional PC, the class's own local `update_wts`
per edge.

## Background (why this evades every prior failure)

The pinned mechanism, each edge's top-down map was only ever fit around bottom-up operating
points, so the composed cascade is off-manifold miscalibrated (content-blind, ~10-20x gain). The
end-to-end calibration contrast fed each weight step the whole 9-layer compounded miscalibration
and destabilized at every non-degenerate expression level. This design changes the credit
assignment, not the objective. During recon (both clamped) every image-path layer's state is an
exact, bounded, per-layer target that ENCODES THE ACTUAL IMAGE. Aligning each edge's top-down
prediction of its cascade input to the recon target one layer below gives every weight step a
one-layer error, bounded by recon magnitudes, never the compounded one. If each edge maps
cascade inputs to recon targets, the composed cascade renders the image by induction.

## This is bidirectional PC

Every phase is clamps plus the model's own relaxation, and every weight change is the class's
own local `update_wts`/`update_b` (the d_pred term IS the per-edge alignment when the layer
below is clamped at its target). One shared-weight net both directions, no backprop, no separate
decoder, no optimizer.

## The step (`cascade_step`, module-level in train_coco64.py, `--train-mode cascade`)

- Phase R (targets), both clamped (recon config), `pass_through(img, txt, mask)`, relax `k0`
  with all layers. Snapshot the state of every image-path layer that has one (the decode
  layers, the latents, and `img_input`, whose target is the clamped true image). The latents
  are now image-set (the well-posed conditioning) and stay held by exclusion.
- Phase C (cascade), WITHOUT re-running pass_through (the latents must keep their image-set
  states), zero the decode states and the image directly (`state.assign(zeros)`), then relax
  the decode top-down-dominant (`pi_bu = 0`, `state_lr` adequate, `k1` steps, the existing
  precision knobs) so the decode states hold the actual current cascade expression from the
  image-set latents. Snapshot these cascade states.
- Phase W (per-edge alignment), for each decode layer L whose `prev_layer` P has a snapshotted
  target, one edge at a time, set `P.state = target[P]` and `P.is_clamped = True` (so
  `update_wts` takes the d_pred-only branch), leave `L.state` at its cascade value, run
  `L.update_wts(); L.update_b()` at `gen_lr` with weight decay off, then RESTORE `P.state` to
  its cascade value and unclamp P (each layer is the L-role for one edge and the P-role for
  another, so set/restore per edge). The bottom edge aligns `conv1.predict_prev(cascade)` to
  the true image. End in the recon clamp config (image+text clamped, everything else unclamped).
- Interleave with recon steps like the other modes (`--gen-every`), warm-start from
  `ckpt_gen_best`, `--weight-norm` on (trust normalizes the per-edge steps).

## Why stability is expected

Each weight step's error is one layer's miscalibration against a bounded recon target (recon
states are ~1e2, not the compounded 10-20x saturation), the trust ratio normalizes step size,
weight decay stays on for the interleaved recon steps, and the recon interleave keeps the
bottom-up operating points anchored. The destabilizing ingredient of the calibration CHL (the
end-to-end anti-learn at saturated states) is absent, there is no anti-learn phase at all, this
is per-edge supervised alignment with the model's own recon trajectory as teacher.

## Validation

- Defaults byte-identical (the new mode is opt-in, no layer-file change beyond what already
  gated). Syntax + a cluster smoke (64 pairs, finite, bounded, TRAIN_DONE, no clamp errors).
- The decisive retest, `latent_source_diag` on the trained checkpoint at the generative
  schedule (`--pi-bu 0 --decode-state-lr 0.25`) and the boost baseline. VERDICT, does
  decode(img-set) turn from content-blind saturation into the recognizable true image (low MSE,
  per-column identity). Success also unlocks fix B (text-to-code alignment) for full text->image.
- Watch training with the corrected grep (energy >= 1, state >= 400, terminal states).

## Risks

- The cascade inputs early in training are the saturated expression, per-edge errors are larger
  than recon-scale until the top edges calibrate, gen_lr starts gentle (1e-4) and the trust
  ratio guards. If it still destabilizes, teacher-forcing the cascade inputs from the recon
  states with an annealed mix is the fallback.
- Edges whose prev is a stateless helper are skipped and calibrate indirectly through their
  neighbors, acceptable for the probe.
- gelu compensation, each edge's alignment must absorb the activation mismatch around cascade
  operating points, exactly what the d_pred term fits.

## Out of scope

Fix B (text-to-code alignment, only after the decode milestone), the isometry repair (stackable
later), held-out, the ladder.
