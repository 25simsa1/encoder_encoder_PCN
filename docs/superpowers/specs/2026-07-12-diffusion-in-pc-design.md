# Design: diffusion-in-PC, a conditional denoiser (Approach B3)

Date: 2026-07-12

## Goal

Beat the mean-collapse for text-to-image by making the bidirectional PC net a CONDITIONAL
DENOISER trained with a local denoising-reconstruction rule, and generating by reverse diffusion.
Because the denoiser is conditioned on a NOISED IMAGE (not only the under-determined latent), its
target is sharp for small noise, and chaining small denoising steps composes a sharp sample.
Strictly bidirectional PC, on the 2k COCO64 overfit. This is the escalation after the deterministic
reweighting (A) and the Langevin EBM (B1) both came back null.

## Background (why this beats what A and B1 could not)

A and B1 failed because the decode target `E[x_0 | latent]` is blurry, the latent under-determines
the image. Diffusion changes the conditioning: the denoiser predicts the clean image from a noised
image `x_t`, and `E[x_0 | x_t]` is SHARP for small noise (the noised image nearly determines the
clean one). The detail is carried by `x_t`, not the latent. Reverse diffusion chains many small,
sharp denoising steps from noise to a crisp image, with the caption conditioning every step.

## This is bidirectional PC

One shared-weight net used both directions: the encoder (image up to latent) sees `x_t`, the
decoder (latent down to image) predicts the clean image. Training is the class's own local weight
step toward a clean target (no backprop, no separate decoder, no optimizer). The diffusion noise is
applied to the image DATA before it is clamped, so `update_state` and the layer code are untouched
and off is byte-identical by construction.

## Component 1: the noise schedule (forward process)

A short schedule of `N` noise levels (default ~10), `sigma_1 < ... < sigma_N`, e.g. geometric from
a small `sigma_min` to a large `sigma_max` (~the image std). Forward process `x_t = x_0 +
sigma_t * eps`, `eps ~ N(0, I)`, clipped to a sane range. Built in `train_coco64.py`; a training
step samples a level, the reverse sampler walks the schedule down.

## Component 2: the denoising training step (local, PC-native)

A module-level `diffusion_step(m, img, txt, mask, k0, k2, gen_lr, sigma_t)`, and
`--train-mode diffusion`. Positive-only recon toward the CLEAN image, reusing the CHL two-phase
machinery:
- ENCODE phase: form `x_t = x_0 + sigma_t*eps`, clamp `img_input = x_t` and the caption, and relax
  ALL states `k0` so the latent (and the whole net) encodes `x_t` and the caption. Then hold the
  latent fixed (unclamped, excluded from the decode loop), exactly as `chl_step` does.
- TARGET phase: set `img_input = x_0` (the CLEAN image) and clamp it, relax the decode `k2` with the
  latent fixed, then take the class's local weight step at +gen_lr so the decode learns
  `latent(x_t) -> x_0`. This is a denoising autoencoder, sharp for small `sigma_t`, blurry for
  large, which is the correct diffusion behavior.
Sample `sigma_t` per step (uniform over the schedule) so the denoiser learns every level. No
contrast / anti-learn phase is needed (this is supervised denoising, not an EBM). Ends in the recon
clamp config.

## Component 3: reverse-diffusion generation (a sampler tool)

`tools/diffusion_sample.py`. Start from pure noise `x_N = sigma_N * eps`, clamp the caption. For
`t = N .. 1`:
- set `img_input = x_t`, clamp it, relax the net `k` steps so the encoder maps `x_t` + caption to
  the latent and the decode predicts; read `x0_hat = conv1.predict_prev()` (the decode's clean-image
  estimate).
- step down a level, `x_{t-1} = x0_hat + sigma_{t-1} * eps` (predict-`x_0`, re-noise to the next
  lower level; the simplest ancestral sampler).
At `t = 0`, `x0_hat` is a sharp, caption-specific sample. Draw a few per caption and save an image
grid; the visual is the judge.

## Component 4: opt-in and flags

- `--train-mode diffusion` selects the denoising step (recon / gen / chl / ebm unchanged).
- `--diff-levels N` (default 10), `--diff-sigma-min` / `--diff-sigma-max` the schedule.
- Off (any non-diffusion mode) is byte-identical, the layers are UNCHANGED by B3 (the noise is on
  the data), so NATIVE (`GATE_MATCH nlayers=143`) and the COCO64 gate hold trivially. Composes with
  `--weight-norm` (keep it on for stability).

## Validation

- The COCO64 gate stays green (no layer change; re-confirm once). NATIVE-143 gate deferred to a big
  GPU as before.
- A stability smoke on COCO64_GEN (`--train-mode diffusion --weight-norm`): does denoising training
  stay finite and bounded?
- The decisive retest: warm-start from the recon best, train the denoiser across levels, then run
  the reverse sampler and save image grids per caption. Do the samples look SHARP and
  caption-specific (recognizable scene structure), and vary across draws, versus the RGB-noise of B1
  and the 0.25-contrast blur of A and CHL? This is a visual judgment; also report brightness/contrast
  for continuity.

## Risks and unknowns

- From-scratch diffusion at 64px on 2k pairs is not guaranteed; the denoiser may under-fit or the
  reverse chain may drift. The short schedule, weight-norm, `x_0`-prediction, and the warm start are
  the guards; the schedule and `k`/`k0`/`k2` are the tuning.
- The cross-modal conditioning (caption steering the denoiser) is the untested part, the denoiser must
  learn to use the caption when `x_t` is mostly noise (early reverse steps). If it ignores the caption,
  generation is caption-agnostic; the fix is to up-weight the caption edge or condition more strongly.
- Predict-`x_0` re-noise is the crudest sampler; if it is unstable, a DDIM-style deterministic step is
  the fallback.
- More machinery than any prior approach; kept minimal (one training step, one sampler, no layer
  change) to stay tractable.

## Sub-decisions (chosen defaults)

- The denoiser predicts `x_0` directly (the clean image is the clamp target, so it drops straight into
  the existing recon machinery), rather than predicting the noise `eps`.
- A short schedule of ~10 levels with a matching number of reverse steps; 2k-overfit sharpening does
  not need hundreds.

## Out of scope

Held-out / generalization, the capacity ladder, learned variance / classifier-free guidance. B3 is the
first PC-native diffusion probe.
