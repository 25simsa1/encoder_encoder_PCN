# Design: contrastive-Hebbian (CHL/EqProp) generative objective for sharp text-to-image

Date: 2026-07-10

## Goal

Teach the shared image decode to produce the true image from the text-driven latent when
run STANDALONE (the model's own top-down generation), so text-to-image yields sharp,
full-amplitude, caption-specific images instead of the dark low-amplitude conditional mean.
Replace the current one-sided generative step (which learns a true-image-assisted bridge)
with a two-phase contrastive-Hebbian update. Stays entirely within bidirectional predictive
coding.

## Background (why this objective)

The diagnostics localized the last gap precisely. The text path and latent are healthy
(text-set latent scale matches image-set), the pathway is reconnected (invertible
downsampling), and the caption drives the image per-pixel under a top-down boost. What
remains: the top-down decode outputs the low-amplitude CONDITIONAL MEAN (dark, blurry,
~0.4x the true brightness). The current generative step (phase 2: freeze the text-driven
latent, clamp the true image at the bottom, relax the decode to bridge, local weight step)
does not sharpen and in fact darkens over steps. The objective diagnostic showed this is not
weight decay (decode weight-norm unchanged) and not the latent; it is that the clamped true
image "helps" during the weight step, so the decode never learns a strong STANDALONE
latent->image map. The fix is to supervise the pure standalone generation against the truth.

## Approach: contrastive Hebbian learning (chosen)

CHL is the classic local learning rule for energy/PC nets, and the full-clamp form of
equilibrium propagation. It contrasts two relaxations of the same net under different
clamping and updates weights by the difference. Here:

- Phase 0: text-drive the latents. Caption clamped, image = zeros unclamped, relax so the
  five shared latents become text-set. This latent is then HELD FIXED (unclamped but excluded
  from the relax, exactly as in the current generative step) through BOTH phases below, so the
  contrast varies only the image, not the latent.
- FREE phase. With the text-set latent held fixed, leave the image FREE (unclamped, zero init)
  and relax the decode intermediates. This is the standalone generation from the text latent
  (today, dark/blob). No true image present.
- CLAMPED phase. With the SAME text-set latent held fixed, clamp img_input to the TRUE image
  and relax the decode intermediates. This is the target: what the decode should produce from
  that same latent.
- Contrastive update on the decode layers. Strengthen the clamped (true-image) configuration
  and weaken the free (generation) configuration: net `wts -= gen_lr * (g_clamped - g_free)`,
  where g is the class's own local weight gradient. Because the latent is IDENTICAL in both
  phases, the update purely teaches decode(text-latent) to turn its free output into the true
  image. It trains gamma=0 standalone generation, not a boosted or true-image-assisted one.

Contrast with the prior step: the prior step only did the clamped-phase learn (and let the
clamped image assist the bridge). CHL adds the FREE-phase contrast, which is what forces the
decode to stand alone.

## Hard constraints (this is bidirectional PC)

- ONE shared-weight net used both directions (predict_next up, predict_prev down). The decode
  is the top-down direction of the same weights, NOT a separate network.
- Both phases are the model's own relaxation (update_state); the ONLY difference between them
  is the image clamp. The true image and caption enter ONLY as clamps, never as a
  differentiated loss.
- The weight update is the class's own LOCAL rule (update_wts, the beta-less LARS on local
  prediction errors), applied at the two equilibria. CHL is local: each layer uses only its
  own neighbors' states at each equilibrium. NO backprop through the net, NO global loss
  gradient, NO Adam/momentum, no change to the per-layer update math.
- The five shared-latent aliases stay. NATIVE_7B untouched (a COCO64_GEN train mode), keeps
  GATE_MATCH nlayers=143.
- Stable recipe (lr 1e-3 recon, wd 3e-2 recon, state_clip 400, gelu on stride-1 convs).

## Implementation (reuse the existing local rule, sign-flipped)

Add a training mode (e.g. `--train-mode chl`) whose per-generative-batch step does:
1. Phase 0 (set the latent): caption clamped, image = zeros unclamped, relax k0 with
   update_state so the shared latents become text-set. Then HOLD the shared latents fixed
   (leave them unclamped but exclude them from all further relax/weight loops, exactly as the
   current generative step does) for the rest of the step, so both phases share this latent.
2. FREE relax: latents fixed, image FREE, relax the decode intermediates k1 with update_state.
3. Apply update_wts/update_b on the decode layers with a NEGATIVE rate (`learning_rate =
   -gen_lr`) and weight_decay 0 (anti-learn the free config). Reuses the negative-lr path the
   `--gen-lr` knob already supports.
4. CLAMPED relax: latents STILL fixed (same as phase 0), img_input clamped to the TRUE image,
   relax the decode intermediates k2 with update_state.
5. Apply update_wts/update_b on the decode layers with a POSITIVE rate (`+gen_lr`),
   weight_decay 0 (learn the clamped config).
6. Restore the recon clamp config and the layers' original learning_rate/bias_lr/weight_decay.

The decode-intermediate set (relaxed and weight-stepped in both phases) is the same one the
current generative step uses: the weight-bearing image-path layers minus the five shared
latents minus img_input. The shared latents are the fixed top-down source in both phases;
img_input is free in the free phase and clamped-to-true in the clamped phase.

Net decode update over the two applies: `wts -= gen_lr*(trust_c*g_clamped - trust_f*g_free)`.
Realization note: because update_wts applies the LARS trust ratio per call, each phase's
gradient is LARS-normalized before the contrast (so it is a LARS-scaled CHL, not textbook
CHL). This keeps the class's own weight-step machinery and is directionally correct (it
reduces energy at the clamped config and raises it at the free config). If the LARS
per-phase scaling proves to hurt, the fallback is a raw-gradient mode on update_wts (return
g without the LARS/wd wrapper) so the plain difference `gen_lr*(g_clamped - g_free)` can be
applied; that touches the layer classes and would be gated, so it is a fallback, not the
first cut.

Interleave with the recon step (recon every batch to keep the encoder/latent/text path/i2t
healthy), warm-start from the stable recon-trained COCO64_GEN checkpoint, gentle gen_lr,
weight decay off for the CHL steps.

## Validation

- NATIVE GATE_MATCH nlayers=143 (chl mode is a COCO64_GEN train path; recon default unchanged).
- CHL stability smoke on COCO64_GEN (a few epochs): both phases + both weight applies run, the
  clamp/lr/wd hygiene restores cleanly (no compiled-sweep guard error), energy and max|state|
  finite, no divergence. CHL can be finicky, so this is the first gate.
- Retrain (warm-started) then re-run the darkness diagnostic + the text-to-image retest. Success:
  the STANDALONE (gamma=0) generation brightens toward the true image (gen/true mean ratio rising
  from ~0.4 toward 1.0) and the images show caption-specific structure, with reconstruction and
  image-to-caption still intact.

## Risks and unknowns

- CHL stability. The free/clamped contrast can be noisy or unstable, especially with the
  LARS-scaled realization; watch energy and max|state|, use a gentle gen_lr, and apply the
  B-gate discipline (stop and consult if it destabilizes).
- The free equilibrium may be a near-blob (input-insensitive) early on, weakening the
  per-caption signal; the per-caption clamped target should still carry it, and as the free
  equilibrium moves off the blob the signal should strengthen.
- Two relaxations plus two weight applies per generative step cost more compute (slower on L4).
- The LARS per-phase normalization is a deviation from textbook CHL; the raw-gradient fallback
  is available if needed.

## Out of scope

The raw-gradient update_wts mode (fallback only), held-out generalization, the capacity ladder,
multi-scale weighting, and any change to reconstruction. Each is a later step if standalone
generation sharpens.
