# Design: high-frequency-weighted PC energy for sharp text-to-image (Approach A)

Date: 2026-07-11

## Goal

Break the mean-collapse that keeps text-to-image blurry, by reweighting the bottom image
prediction error to emphasize high frequencies, so the decode's local learning is forced to
reproduce edges and detail instead of settling on the smooth conditional mean. Strictly within
bidirectional predictive coding. Fast, cheap probe first; if it plateaus, escalate to the
stochastic-sampling mechanism (Approach B, out of scope here).

## Background (why generation is blurry)

Squared-error PC decode training is minimized by the conditional mean `E[x | latent]`. High-
frequency detail contributes almost nothing to a squared pixel error, so the decode is free to
drop it and the output is a low-amplitude blur. The weight-norm retest confirmed this directly:
with training now STABLE past ep13, the CHL decode still converged to the conditional mean and
got FLATTER with more training (boosted-gen contrast 0.404 at ep1 to 0.243 at ep12; brightness
stuck at ~0.40) even as the text-path latent drive improved. So the objective, not the
stability, is the ceiling. On the 2k overfit each caption maps to one specific image, so forcing
the decode to reduce high-frequency error should push it to memorize the sharp detail it now
averages away.

## Where the bottom image error lives

`conv1` (`Conv2DPCNLayer`, `encoder_encoder_pcn.py:164`) has `prev_layer = img_input` (the pixel
layer). Its `predict_prev()` decodes `conv1.state` back to pixel space, and its `update_wts`
d_pred term forms the pixel error `act(conv1.predict_prev()) - act(img_input.predict_next())`
(shape (B, 64, 64, 3)) and feeds it to `Conv2DBackpropFilter`. That pixel error is the single
place the decode weights learn how well the reconstruction matches the image, so it is where the
high-frequency weighting belongs.

## Core mechanism: high-pass boost of the bottom error

Reparameterize the bottom pixel error `e = act(pred) - act(prev.predict_next())` used in
`conv1.update_wts` as

  e' = e + gamma * HP(e)

where `HP` is a FIXED high-pass spatial filter (a 3x3 Laplacian applied per channel via
depthwise conv, `HP(e) = e - blur(e)` equivalently) and `gamma >= 0` is the boost strength. This
up-weights the high-frequency component of the error by `(1 + gamma)` while leaving the low-
frequency component at weight 1, so the decode weight step reduces high-frequency error harder
and learns sharp detail. `gamma = 0` gives `e' = e` exactly (byte-identical to today).

This is bidirectional PC. `HP` is a fixed linear filter, so `e'` is still the layer's own local
prediction error (reshaped), and the weight update is still the class's local `update_wts` with
no backprop, no separate decoder, no optimizer. It changes only WHICH part of the error the
local rule emphasizes.

## Scope and opt-in

- A per-layer `hf_gamma` float on `Conv2DPCNLayer` (default 0.0). Only the bottom conv (the one
  whose `prev_layer` is `img_input`, i.e. `conv1`) gets it set; every other conv keeps 0.0. When
  `hf_gamma == 0` the branch is skipped and the layer is byte-identical.
- A fixed Laplacian kernel built once on the layer (a `(3,3,C,1)` depthwise high-pass, reflect or
  SAME padding to preserve shape). The reweighting wraps the existing `(act(pred) -
  act(prev.predict_next()))` expression in each activation branch of `conv1.update_wts`'s d_pred
  before it enters `Conv2DBackpropFilter`.
- A training flag `--hf-weight GAMMA` in `train_coco64.py` (default 0.0) that sets `hf_gamma`
  on the bottom conv after the model is built. Off = byte-identical, so NATIVE and the COCO64
  gate are unaffected. Applies during both the recon and generative/CHL weight steps (the decode
  learns sharp in both regimes, which should transfer to text-driven generation).

## This composes with weight-norm

The weight-norm stabilizer stays available and independent (`--weight-norm`). The HF weighting
changes the error direction, not the norm control. A run can use both (stable AND sharp-seeking)
or HF alone. The bottom conv's `update_wts` already routes through `weight()`, so the two flags
compose with no extra work.

## Validation

- NATIVE `GATE_MATCH nlayers=143` (and the COCO64 gate) with `--hf-weight 0` OFF: byte-identical,
  since the `hf_gamma == 0` branch is skipped. Proves the change is inert when off.
- A local unit test: with `hf_gamma = 0`, `conv1.update_wts` produces the same weight delta as
  before; with `hf_gamma > 0`, the delta differs and the high-pass of a smooth (low-frequency)
  error is ~0 (so a smooth error is barely reweighted) while a sharp (edge) error is boosted.
- A short stability smoke on COCO64_GEN with `--hf-weight` at a small gamma (e.g. 0.5, 1.0):
  does training stay finite and bounded (HF weighting amplifies high-frequency error, which
  could destabilize; pair with `--weight-norm` and `--state-clip` if needed)?
- The decisive retest: retrain the 2k overfit (recon, then the generative/CHL step) with
  `--hf-weight`, and run `darkness_diag`. Does the boosted-generation contrast (std ratio) rise
  meaningfully above the 0.24-0.40 plateau toward the true image's, and do the generated PNGs
  show sharper, more caption-specific structure, with brightness moving off the ~0.40
  conditional-mean floor? Sweep gamma (0.5, 1, 2) to find the useful range.

## Risks and unknowns

- High-pass weighting amplifies high-frequency error and could destabilize training or inject
  high-frequency artifacts rather than true detail; the gamma sweep and pairing with weight-norm
  are the guards.
- For the OVERFIT this forces sharp memorization; it will not by itself give sharp GENERALIZING
  generation (that is the sampling story, Approach B). This probe is scoped to the overfit.
- The high-pass emphasizes edges; it may raise contrast without fixing overall brightness (the
  conditional-mean amplitude). If brightness stays at ~0.40, that points at Approach B/C next.
- Choosing to reweight only the decode weight step (not the pixel state relaxation) keeps the
  generation-time relaxation unchanged; if the sharpening does not appear at generation, a
  follow-up is to also HF-weight `img_input`'s state update.

## Out of scope

Approach B (stochastic sampling) and C (self-encoded feature matching); generalization/held-out;
the capacity ladder. This is the fast deterministic probe of whether breaking the amplitude
helps at all.
