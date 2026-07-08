# Design: invertible (bidirectional) image downsampling for text→image generation

Date: 2026-07-07

## Goal

Unblock text→image spatial generation in the bidirectional PC class by making the
image path's downsampling **bidirectionally invertible**, so top-down (generative) drive
can flow through it. Replace the one-way max-pooling with stride-2 shared-weight
convolutions (`predict_next` = strided conv2d down, `predict_prev` = transpose conv2d up,
same weights). Config-driven and opt-in: `NATIVE_7B` keeps max-pool and stays
byte-identical (`GATE_MATCH nlayers=143`); a new `COCO64_GEN` config uses the strided-conv
downsamplers. Retrain on the COCO64 2k overfit and check whether text→image now produces
caption-varying structure.

## Background (why this fix — the localized mechanism)

A chain of read-only diagnostics on `ckpt_gelu_best` localized the text→image failure:
- The text path is healthy: distinct captions give distinct text codes (`inter12` PR 7.4)
  and set a diverse shared latent (`dense2` PR 7.7).
- The decode weights preserve that diversity (`dense2.predict_prev()` PR 7.15).
- But the generated image code collapses (`inter2` PR 0.72) and the image is a uniform
  field, because **top-down generative drive is structurally blocked at every maxpool**:
  `MaxPool2DPCNLayer` is `is_clamped=True` with no `state` and no `predict_prev` (no
  unpool), and every layer's top-down relaxation block skips clamped next-layers
  (`if layer.is_clamped: continue`). So a conv below a pool never receives top-down
  through it — the text-set latent's drive reaches only the layers above the topmost pool
  (dense1/inter1/conv9); the lower conv stack (spatial detail) sees only bottom-up from
  the zero image and collapses. Reconstruction and image→caption work because they don't
  need top-down through the pools (image drives bottom-up; the text path has no pooling).

Root cause: max-pool has no PC-consistent top-down inverse, so the image path is
bidirectional only above the first pool. The fix is to make the downsampling a shared-weight
conv, which has a natural transpose — restoring bidirectionality through the pools. This is
Approach A from brainstorming (strided conv), chosen over average-pool+uniform-unpool
(parameter-free but blurry/fixed) and max-unpool-with-indices (breaks for generation: the
argmax indices require a forward pass of the image being generated, and the inverse is sparse).

## Hard constraints (inherited + this feature)

- The bidirectional class only; ONE shared-weight image net used both directions. The
  strided downsampler uses the SAME weights forward (`conv2d`) and top-down
  (`conv2d_transpose`). NO separate decoder, NO backprop through the net; weight learning
  stays the existing local beta-less LARS.
- The five shared-latent aliases stay. All existing conv output sizes, the flatten→dense
  projections, and the 5 multi-scale taps (off conv2/4/6/8/9) stay byte-identical in shape.
- Config-driven / opt-in: `downsample` defaults to `'maxpool'`; with the conv `stride`
  defaulting to 1, `NATIVE_7B` (and the existing `COCO64_156M`) are byte-identical and
  `NATIVE_7B` still `GATE_MATCH`es at `nlayers=143`.
- Runs on the H200 via `tools/clusterrun.sh`; commits first-person student, no AI attribution.
- Training uses the stable recipe (lr 1e-3, weight_decay 3e-2, state_clip 400, gelu conv);
  watch the known norm-inflation instability.

## Component 1: strided-conv support in `Conv2DPCNLayer`

Add a `stride: int = 1` attribute (constructor arg, default 1). Thread it through every
spatial op so a stride-2 conv is fully bidirectional:
- `net_in` / `__call__`: forward `conv2d(..., strides=self.stride, ...)`.
- `predict_prev`: `conv2d_transpose(..., strides=self.stride, output_shape=...)`, where the
  transpose `output_shape` reconstructs the PRE-downsample spatial size. Under SAME with
  stride s, the forward maps H → ceil(H/s); the transpose must expand ceil(H/s) → H (the
  stored input H from the forward `output_shape`/`net_in` input). The layer must record the
  input spatial size at forward time so the transpose can restore it exactly.
- `pred_loss_d_input`: the transpose there also uses `strides=self.stride` and
  `output_shape=x.shape` (already shape-driven by `x`).
- `update_state`: the bottom-up `tf.nn.conv2d(..., strides=self.stride, ...)` term.
- `update_wts`: `tf.raw_ops.Conv2DBackpropFilter(..., strides=[1, s, s, 1], ...)`.

Default `stride=1` leaves every call byte-identical to today (NATIVE unaffected — proven by
the gate). The transpose `output_shape` for stride 1 is unchanged from the current code.

## Component 2: config

`pcn_config.py`: add `downsample: str = 'maxpool'` to `PCNConfig` (with a `__post_init__`
check that it is one of `{'maxpool', 'strided_conv'}`). `NATIVE_7B` and the existing
`COCO64_156M` keep `'maxpool'` (prior results reproducible). Add a new config
`COCO64_GEN` = `COCO64_156M` with `downsample='strided_conv'`.

## Component 3: constructor branch

`encoder_encoder_pcn.py`: at each of the 4 downsample points (after conv2, conv4, conv6,
conv8), build EITHER a `MaxPool2DPCNLayer((2,2), prev)` (when `downsample=='maxpool'`) OR a
stride-2 downsampler `Conv2DPCNLayer(prev_channels, (2,2), lr, 'linear', prev,
padding='SAME', stride=2)` (when `'strided_conv'`). Channels are preserved (in=out=the
preceding conv's channel count), so the following block's conv input is unchanged. Wiring is
identical to the maxpool case: the pre-pool conv's `next_layers` still also carries its
flatten tap; the downsampler's `prev_layer` = the pre-pool conv, `next_layers` = [next
block's first conv]. A `strided_conv` downsampler is appended to `trainable_layers` (it has
weights); a maxpool is not (unchanged). Helper builds the layer so the four sites stay DRY.

## Data flow

Encode (bottom-up): the strided conv halves H,W exactly as the pool did — all downstream
feature-map sizes, flatten dims, dense projections, and taps unchanged. Generate (top-down):
the text-set latent's drive propagates conv9 → downsampler4 → conv8 → … → conv1 → img_input
through each downsampler's `conv2d_transpose` (the downsampler is a real, unclamped conv, so
the relaxation's top-down block no longer skips it). The severed generative pathway is
reconnected end-to-end.

## Component 4: retraining + evaluation

- Train `COCO64_GEN` from scratch on the 2k COCO64 overfit with the stable recipe (lr 1e-3,
  wd 3e-2, state_clip 400, gelu conv), logging energy + max|state|; checkpoint lowest-energy.
- Text→image retest (150 relax, in-sample pairs): the KEY check — does text→image now produce
  caption-VARYING, recognizable STRUCTURE (generated-image participation ratio well above 1;
  PNGs that differ by caption and show scene content, not the uniform field)? Reconstruction
  and image→caption must stay intact. Energy must still descend (no divergence).

## Validation / gates

- `NATIVE_7B` `GATE_MATCH nlayers=143` with `downsample='maxpool'` (default) + conv
  `stride=1` (default) — proves the stride plumbing and the config field are inert on the
  original model.
- `COCO64_GEN` builds and is structurally sound: spatial progression matches the maxpool
  version, `SHARED_STATE_ALIASES=5`, both generation directions finite, batched relaxed step
  finite, param count ≈ `COCO64_156M` + ~1.3M (the four 2×2 stride-2 convs).
- Local unit test: the strided conv's forward→transpose shape round-trip returns the original
  (H, W, C) — down then up recovers the input spatial size.

## Risks and unknowns

- The fix reconnects the pathway, but the shared 100-dim latent may still bottleneck spatial
  detail; text→image could improve yet remain coarse. In-sample structure is the first,
  achievable check.
- Retraining reintroduces the norm-inflation instability; the stable recipe mitigates but the
  new downsampler weights change the dynamics — watch energy/max|state|.
- Transpose-conv can produce checkerboard artifacts; 2×2 kernel first, 4×4 is a later knob.

## Out of scope

Held-out generalization, multi-scale InfoNCE, average-pool / max-unpool variants,
anti-checkerboard kernel tuning, and pointing `train_step` (interleaved) at the relaxed
schedule. Each is a later step if in-sample text→image structure appears.
