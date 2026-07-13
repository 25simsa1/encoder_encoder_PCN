# Design: untied top-down prediction weights (the constraint amendment)

Date: 2026-07-13

## Goal

Remove the proven root of the generation failure by giving each image-path edge its OWN
top-down prediction weights `wts_td`, trained by the same local rule, so the bottom-up and
top-down duties no longer share one matrix. Opt-in, byte-identical off, from-scratch recon,
then the standard retest.

## The constraint amendment (recorded)

The project rule was one shared weight both directions. The research lead amended it on
2026-07-13 after the campaign pinned the shared weight as the root, eight mechanisms and three
inference routes all failed on the tug-of-war (any objective pulling the shared matrix toward
top-down duty is either out-anchored by recon or rips recon apart) and on the transpose
non-invertibility (GELU and the strided downsampling are inverted by no transpose). What STAYS,
one network with reciprocal connections, relaxation inference, local prediction-error learning
only, no backprop through depth, no optimizer, no separate decoder module (the top-down weights
live on the same edges). Separate feedback weights learned locally are standard in the PC
family; tying was the stronger assumption. The tied-vs-untied comparison becomes the paper's
controlled ablation, we proved tying is the blocker, then removed exactly that.

## Why this dissolves the pinned root

`update_wts` already computes two local gradients with different meanings, `d_state` (the
bottom-up prediction error, the encoder's duty) and `d_pred` (the top-down prediction error,
the decoder's duty). Today they are summed into ONE matrix, the tug-of-war. Untied, `d_state`
trains `wts` and `d_pred` trains `wts_td`, each with its own LARS step, nothing competes, and
the top-down map can learn a genuine content-carrying decode (it is no longer a transpose of
anything, so the non-invertibility of the transpose is moot).

## Mechanism (Conv2DPCNLayer and DensePCNLayer)

- `self.wts_td = None` default. `weight_td()` accessor, returns the effective top-down weight,
  `wts_td` when untied else the tied `weight()` (byte-identical off).
- `enable_untied()`, requires realized `wts`, creates `wts_td = tf.Variable(tf.identity(wts))`
  (seamless, predictions unchanged at enable time).
- Routing, `predict_prev` and the top-down error terms in `update_state` (the prev-block
  `d_pred` that propagates the top-down error into the state) use `weight_td()`. `net_in` and
  `pred_loss_d_input` (the true gradient of the bottom-up prediction error) stay on the tied
  `weight()`. The bias `b` stays shared (minor, acceptable).
- `update_wts` when untied, TWO independent local steps with the class's own LARS trust,
  `wts -= lr * trust(d_state) * (d_state + wd*wts)` and
  `wts_td -= lr * trust(d_pred) * (d_pred + wd*wts_td)`; tied path byte-identical.
  Note `d_pred` as computed measures the top-down error through `predict_prev`, which now reads
  `wts_td`, so it IS the local gradient for `wts_td` (same shapes, same ops).
- `--untied` in train_coco64.py, enable on the image-path conv/dense layers after weights are
  realized (and after any resume-restore); `wts_td` persisted in a side checkpoint
  `<ckpt>_td` mirroring the weight-norm pattern, ALL_W untouched.

## The run and the retest

From-scratch COCO64_GEN recon, 15ep, stable recipe (lr 1e-3, wd 3e-2, clip 400, gelu), one
change at a time (no weight-norm, no isometry in the first run). During recon the `d_pred`
steps now train `wts_td` at every edge to decode the bottom-up states, the per-edge inverse
learned WITHOUT opposition, which is the thing that never existed before. Then the standard
retest on the result, latent_source_diag at pi(0, 0.25), plain(1.0, 0.2), boost(gamma 1), does
decode(img-set) finally carry identity. If yes, fix B (text-to-code alignment) completes
caption to image.

## Validation

- Unit tests, off is byte-identical (weight_td falls back, update_wts single-step unchanged),
  enable is seamless (predict_prev unchanged at enable), and after training steps with
  divergent duties the two matrices actually diverge while each error falls.
- COCO64 gate vs the banked ref with the flag off (tol 1e-4 on L4, the 2e-4 envelope on HEQ).
- Training telemetry as usual, plus the decisive retest above.

## Risks

- Doubles the image-path weight memory (fine at COCO64 scale, a consideration for NATIVE later).
- The top-down stack may still need a generation-time schedule (boost or precisions) to express;
  the retest covers all three.
- Norm inflation on `wts_td` (its own LARS); the stable recipe held recon-only runs ~ep24, and
  weight-norm/isometry can be added later if it drifts.

## Out of scope

Text-path untying (text generation is not the goal), fix B (after the milestone), NATIVE
adoption, the ladder.
