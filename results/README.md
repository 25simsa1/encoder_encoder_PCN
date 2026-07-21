# Results (generation study, 2k COCO64 overfit)

Curated milestone grids from the generation line. Each grid: row 1 = true images,
row 2 = decode from image-set latents, row 3 = decode from text(caption)-set latents,
rows 4+ = per-scale latent swaps. Decode is the untied bidirectional-PC relaxation with
an rms-matched readout (`pi` = free precision relaxation, `boost` = single-chain boost).

The rest of the raw decode grids stay local (gitignored) to keep the repo light.

## The recipe behind these
`COCO64_WIDE` (inter_dim 512) encoder trained by local PC recon (lr 1e-3, isometry),
then the top-down decode edges solved in closed form per edge (`tools/ridge_td.py
--ridge-conv`, the fixed point of the local top-down prediction-error rule). Still one
network, reciprocal edges, relaxation inference, local + backprop-free learning.

## Files

- `img2img_wide_boost.png` — the headline. Image-to-image generation with the wide
  pipes + all-edge closed-form ridge. Per-image structure (each decode tracks its true
  image's luminance layout and dominant color), mse 0.0402 vs the 0.069 mean-predictor
  bar. Coarse (64px, low-frequency, checkerboard from the strided transpose), not sharp
  objects. First per-image-structured generation of the project.
- `img2img_wide_pi.png` — same model, free-precision relaxation readout instead of the
  boost. Full contrast, per-image variation, same coarse ceiling.
- `inter100_ridge_ceiling.png` — the earlier inter_dim=100 model with exact closed-form
  optimal decode edges. Still collapses to a shared attractor: the proof that the 100-dim
  inter hourglass was the binding constraint (motivating the wide config above).
- `caption_coupling_attempt.png` — caption-to-image after trunk-level InfoNCE coupling.
  Image-set row keeps its structure; the text(caption)-set row is a flat template. Caption
  to image stays blocked: the text representation does not carry the image-latent values
  (the cross-modal coupling failure this project studies).
