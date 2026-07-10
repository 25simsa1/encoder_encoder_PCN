# Design: PC-native generative-training objective for text-to-image

Date: 2026-07-10

## Goal

Teach the shared image-decode weights to produce the true image from a text-driven
shared latent, so that text-to-image generation yields recognizable, caption-varying
scenes instead of the checkerboard/speckle artifacts the current both-clamped training
produces. The objective must stay entirely within bidirectional predictive coding. It
adds no new network, no separate decoder, and no backprop through the stack. It is
expressed only as a schedule of which states are clamped, plus the model's own
relaxation and its own local weight rule.

## Background

Diagnostics localized the text-to-image failure precisely (see the 2026-07-08 and
2026-07-09 LOG entries). Invertible strided-conv downsampling reconnected the generative
pathway, and a top-down-authoritative generation schedule at inference makes the caption
drive the image per-pixel (image participation ratio rose from 0 to 6.93 as the top-down
boost increased). But the images are artifacts, not scenes, because the decode was
trained only in the both-clamped configuration.

Why both-clamped does not already train text-to-image. Both-clamped training is
symmetric supervised learning of the caption-image pair, but the shared latent, set by
both modalities at once, comes out image-dominated (the clamped image is a strong,
high-dimensional constraint). The decode weights therefore learn to reconstruct the
image from an image-dominated latent. At generation the latent is text-only, which is
off-distribution for those weights. The fix must train the decode weights specifically
on a text-driven latent. The tension is that clamping the true image to obtain a target
re-dominates the latent, returning to both-clamped. Resolving that tension is the design.

## Hard constraints (inherited plus this feature)

- Bidirectional PC only. One shared-weight net used both directions via the same
  predict_next (up) and predict_prev (down). The image decode is the top-down direction
  of the same weights, not a second network.
- Learning stays the existing local rule. Weight updates use the current update_wts and
  update_b (the beta-less LARS on local prediction errors). No backprop through the net,
  no autograd loss, no Adam or SGD, no change to the per-layer update math.
- The true image and the caption enter only as clamps (boundary conditions on the PC
  energy), never as a differentiated loss.
- The five shared-latent aliases stay intact. NATIVE_7B is untouched (this is a
  COCO64_GEN training mode), so NATIVE keeps GATE_MATCH nlayers=143.
- Runs on the cluster via tools/clusterrun.sh or a detached sbatch. Commits are
  first-person student, no AI attribution. Stable recipe (lr 1e-3, weight_decay 3e-2,
  state_clip 400, gelu on stride-1 convs, downsamplers linear).

## Component 1: training structure

Two step types interleaved on the same COCO64_GEN model.
- Recon step, unchanged. The existing both-clamped relaxed schedule
  (update_states_wts_b_relaxed), compiled and fast. Keeps the encoder, the shared
  latents, the text path, and image-to-caption healthy.
- Generative step, new and eager. Trains the image-decode intermediates to produce the
  true image from a text-driven latent.

A training flag selects the mode. `--train-mode recon` (default) is exactly today's
behavior. `--train-mode gen` interleaves a generative step with each recon step. A
`--gen-every N` knob runs one generative step per N recon steps if the two objectives
need rebalancing (default 1, one generative step per recon step).

## Component 2: the generative step

Per generative batch of (image, caption, mask).

Phase 1, text-drive the latents.
- Clamp the caption (txt_input), unclamp the image (img_input), initialize the image to
  zeros, and relax K1 steps with the model's own update_state sweep. Because the image is
  a zero init and unclamped, the five shared latents are set by text alone. This is the
  same relaxation the class already uses for generation in test_step.

Phase 2, supervise the decode against the true image.
- Freeze the five shared latents WITHOUT clamping them. Leave them unclamped but exclude
  them from the relaxation and the weight step, so their states stay at their text-driven
  values (a state changes only when its own update_state is called) while still acting as
  top-down sources. This is the critical detail. Clamping the latents would be wrong,
  because update_state skips clamped next-layers in its top-down block (`if layer.is_clamped:
  continue`), which would remove the latent's downward drive into the decode. An unclamped
  latent that we simply do not relax is both fixed and still driving.
- Clamp img_input to the true image (is_clamped True, set_state to the real image). It acts
  as the bottom boundary and drives the layer above via the bottom-up term.
- Relax K2 steps over the image-path intermediates only (the weight-bearing image layers
  minus the five latents minus img_input). They settle to bridge the fixed text-driven
  latents above and the true image below.
- Take the local weight step (update_wts then update_b) on those same intermediates. The
  local rule reduces each layer's own prediction error given its neighbors, which bakes the
  text-latent-to-true-image bridge into the shared decode weights. The latents are fixed
  top-down sources and are not weight-stepped here (their defining weights are trained by
  the recon step).

Clamp hygiene. The generative step is eager, so its internal clamp changes never reach the
compiled recon sweep. Note that phase 2 already ends in the recon clamp configuration
(image clamped, text clamped, latents unclamped), the same signature the compiled sweep
traced, so the only thing to undo is phase 1's temporary image unclamp, which phase 2 does
by re-clamping img_input. Re-assert image-and-text-clamped, latents-unclamped at the end so
the next compiled recon step's clamp-signature guard is satisfied.

## Component 3: layer sets and how the model exposes them

The generative step needs three sets, exposed by the constructor the way it already
exposes _infonce_codes.
- The five shared-latent pairs, to identify the latents held fixed in phase 2. Derive
  after the full build as the pairs (L.share_state_layer, L) for every layer L whose
  share_state_layer is set. This yields the five (image_dense, text_dense) pairs
  (dense2/dense4, dense6/dense8, dense10/dense12, dense14/dense16, dense18/dense20).
- The image-path layers. The constructor snapshots the image side by recording
  list(self.trainable_layers) at the point just before txt_input is built (all image-side
  layers, including the five image shared latents, are appended before the text path).
  Store as self._image_path_layers.
- img_input, already exposed, to clamp to the true image.

The phase-2 relax-and-weight-step set is the weight-bearing image-path layers (those with
update_wts) minus the ten latent members (both sides of the five pairs) minus img_input.
Those latents are left unclamped but out of the relax and weight loops (fixed top-down
sources), and img_input is the clamped bottom boundary. Structural layers without weights
(the flatten layers) fall out of the set via the update_wts guard.

## Component 4: eager and compiled interleave

The recon step stays on the compiled relaxed sweep for speed. The generative step is
eager (it changes clamp flags mid-step, which the compiled sweep forbids). The training
loop runs the compiled recon step, then the eager generative step, then restores the
recon clamp configuration. If the compiled-plus-eager interleave proves fiddly in
practice, an all-eager fallback (both steps eager) is acceptable at a speed cost, since
the 2k overfit is small.

## Component 5: the B fallback (soft-nudge), gated

If Approach A destabilizes (energy climbing, states pinned at the clip, reconstruction or
image-to-caption degrading, or the run diverging), the fallback is Approach B, a
soft-nudge EqProp-style variant. B keeps the caption clamped and the image unclamped and
adds a weak pull on the image toward the true image during a single relaxation (a nudge
rather than a hard clamp), then takes the local weight step. B is a separate, later design
and MUST NOT be run without checking in with the user first. If A destabilizes, stop,
report, and get explicit approval before implementing or running B.

## Component 6: validation

- NATIVE unaffected. Re-gate GATE_MATCH nlayers=143 (the generative mode is a COCO64_GEN
  training path and does not touch NATIVE or the recon-only default).
- Generative-mode smoke on COCO64_GEN, a few epochs. Both step types run, the clamp
  hygiene holds across the interleave (no compiled-sweep guard error), states are finite,
  energy is sane, no divergence.
- Retrain and run the text-to-image retest (150 relax, in-sample pairs, the model's own
  test_step, no manual boost since the model should now generate). Success is
  caption-varying recognizable structure (image participation ratio well above 1, PNGs
  that differ by caption and show scene content, not speckle), with reconstruction and
  image-to-caption still intact and training stable.

## Risks and unknowns

- The recon and generative steps pressure the shared latent in opposite directions. This
  is the coupling we are forcing and also the main destabilization risk. Watch energy and
  max|state|. If it breaks, the B-gate applies.
- Norm inflation, the standing failure mode. The stable recipe mitigates but the new step
  changes the dynamics.
- The shared 100-dim deepest latent may bottleneck fine spatial detail, so text-to-image
  could become recognizable yet coarse. In-sample recognizability is the first bar.
- Phase 2 trains the decode intermediates while the recon step owns the latent-defining
  weights. If that division starves some decode weight, widen the phase-2 weight-step set
  in a follow-up.

## Out of scope

Approach B itself (gated, separate), held-out generalization, the capacity ladder,
multi-scale or per-scale weighting of the generative signal, and any inference-time
top-down generation schedule (the goal here is that the model's own unassisted test_step
generates). Each is a later step if in-sample recognizable text-to-image appears.
