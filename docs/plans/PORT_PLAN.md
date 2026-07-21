# PORT PLAN — bidirectional Option-1 method onto the 7.7B encoder_encoder_PCN

Planning document only. No model code is changed here, nothing is run, no pod is used. This is for
review BEFORE any implementation. It folds in the four lessons that each caused a real, reproduced
failure in the staged miniatures (Stage 0/1 in `PCN_FIX_PLAN.md`, Stage 1.5 in
`stage1/stage1_5_deep_option1_pcn.py`, Stage 1.6 in `stage1/stage1_6_bidirectional_pcn.py`).

Writing-style note. No colons in prose, no em dashes, per the repo owner's preference.

---

## 0. What the big model actually is (so the mapping is concrete)

From `encoder_encoder_pcn.py`.

- Image branch. A 9-layer VGG-style conv tower with VALID padding and 4 maxpools,
  `conv1(64) conv2(64) mp1, conv3(128) conv4(128) mp2, conv5(256) conv6(256) mp3, conv7(512)
  conv8(512) mp4, conv9(1024)`, on a 572x572x3 input.
- Image reconstruction heads. Off `conv9, conv8, conv6, conv4, conv2` there is a
  `flatten -> Dense(100) "inter" bottleneck -> Dense(huge, relu) -> Dense(100) -> Dense(huge)` chain.
  These huge dense layers (`dense2=102400, dense6=161817, dense10=345871, dense14=702332,
  dense18=1429912`, plus the relu mid-layers `dense1=307200 ... dense17=5433667`) are the generative
  image decoders. The single largest matrix is `inter9 = flatten(conv2) -> Dense(100)`, whose weight
  is roughly `20.6M x 100 = ~2.06B params ~ 8.2 GiB` because `conv2` is at 568x568x64. These heads are
  about 6B of the 7.7B params.
- Text branch. `txt_input -> Dense(512) embedding -> positional encoding -> a transformer PYRAMID`,
  `transformer1..3 (d_model 512) -> linear_1(1024) -> linear_2(48)`, then `transformer4..6 (1024) ->
  linear_3(2048) -> linear_4(12)`, then `transformer7..9 (2048) -> linear_5(4096) -> linear_6(3)`,
  then `transformer10..17 (4096)`. The `linear_2=48 / linear_4=12 / linear_6=3` projections resize the
  SEQUENCE dimension across scales. Each `TransformerPCNLayer` is KQV + softmax attention + add-norm +
  FFN + add-norm.
- Text reconstruction heads. Symmetric `flatten -> inter -> dense` chains off the transformer pyramid.
- The cross-modal coupling, which is the heart of the model. FIVE `share_state_layer` ties make a
  text-side dense and an image-side dense literally share one state Variable.

  | shared latent | image side (from) | text side (from) | state dim |
  |---|---|---|---|
  | S1 | dense2  (conv9 head)  | dense4  | 102400 |
  | S2 | dense6  (conv8 head)  | dense8  | 161817 |
  | S3 | dense10 (conv6 head)  | dense12 | 345871 |
  | S4 | dense14 (conv4 head)  | dense16 | 702332 |
  | S5 | dense18 (conv2 head)  | dense20 | 1429912 |

  So the model is a bidirectional image<->text autoencoder coupled at 5 scales. There is NO one-hot
  label. The anti-collapse anchor is the clamped real input being reconstructed (real magnitude),
  which is exactly the Stage 1 anchor lesson satisfied by reconstruction rather than by a label.

- Current training math (to be removed). `train_step -> pass_through -> update_states_wts_b`, which
  loops over `trainable_layers` calling the hand-written `update_state / update_wts / update_b` per
  layer. Those use the `d_gelu` GELU derivative on a relu forward, an ad-hoc child-averaging in
  `update_state`, and a LARS trust-ratio rescale in `update_wts`. Weights are `tf.Variable(trainable
  = False)` and updated by hand. This is the broken thing the port replaces.

---

## The four non-negotiable lessons and how each maps here

- L1 (Stage 0/1). ONE scalar energy F, every state and weight update is `tape.gradient(F, .)`. The
  hand-written PC math is deleted, not edited. On the big model the anti-collapse anchor is the
  reconstruction of the clamped image and text (real-magnitude targets), so F cannot collapse to 0.
- L2 (Stage 1.6, the silent bug). Any sub-network you want trained must be computed INSIDE the
  learning tape. The relaxation tape holds weights fixed and frees states; the learning tape holds
  states fixed and recomputes ALL trainable modules (conv tower, the 17-block transformer pyramid,
  both decoder stacks) so they get gradients. Get this wrong and the transformer silently freezes
  while accuracy still looks plausible. This is the single highest-risk implementation detail.
- L3 (Stage 1.5). Depth holds only with dense per-scale anchors. The 5 `share_state` scales become
  the per-scale anchors. Every conv layer and transformer block must sit within one hop of an
  anchored scale, or the deep layers freeze (a 400x gradient spread was measured when only the top
  was anchored).
- L4 (Stage 1.6, the precision inversion). For bidirectional reconstruction the generative precision
  must be greater than or equal to the discriminative/coupling precision, the opposite of the bPC
  default. On the big model that means the image-reconstruction and text-reconstruction error terms
  are weighted at least as high as the cross-modal coupling terms, or text->image collapses.

---

## 1. DELETE vs REWRITE (per file)

Principle. Keep every FORWARD and INIT method and the whole graph wiring (it defines the
architecture, which we do not change). Delete every hand-written PC update and hand-derivative.
Rewrite the execution (the train and test loops) around one energy and two tapes.

### `dense_pcn_layer.py`
- DELETE `update_state`, `update_wts` (including the LARS trust step and `last_trust`), `update_b`,
  `pred_loss_d_input`, `d_gelu`, `predict_prev`. Also delete the symptom-patch fields added during
  pod experiments, `state_lr`, `bias_lr`, `state_clip`.
- KEEP `init_params`, `get_kaiming_gain`, `net_in`, the forward in `__call__` (the `net_act`
  computation), `share_state_layer` wiring.
- CHANGE `wts` and `b` to `trainable = True` so the learning tape picks them up via
  `model.trainable_variables`. The state Variable stays `trainable = False` (states are relaxed by an
  explicit GD on F, not by the optimizer).
- The `set_state` side effect inside `__call__` is removed from the forward. State assignment becomes
  explicit in the relax loop.

### `conv_pcn_layer.py`
- DELETE `update_state`, `update_wts` (LARS), `update_b`, `pred_loss_d_input`, `d_gelu`,
  `predict_prev` (the `conv2d_transpose` used only for hand-PC), `state_lr/bias_lr`.
- KEEP `init_params`, `net_in` (the `conv2d` forward), forward `__call__`, `get_kaiming_gain`. KEEP
  `MaxPool2DPCNLayer` forward.
- CHANGE kernel to `trainable = True`.

### `transformer_pcn_layer.py`
- DELETE `AddNormalizePCNLayer.update_state / update_wts / update_b`, and
  `PositionalEncodingLayer.predict_prev / pred_loss_d_input`.
- KEEP all forwards, `AttentionPCNLayer.__call__` (KQV + softmax), `AddNormalizePCNLayer.__call__`,
  `TransformerPCNLayer.__call__`, `PositionalEncodingLayer.__call__`.
- CHANGE all transformer weights to `trainable = True`.

### `encoder_encoder_pcn.py`
- KEEP the entire `__init__` graph construction (layers, `prev_layers/next_layers`, the 5
  `share_state_layer` ties). This is the architecture and it does not change.
- DELETE `update_states_wts_b`, `update_states_wts_b_relaxed`, `update_states_img`,
  `update_states_txt`, and `InputPCNLayer.update_state`.
- REWRITE `pass_through / pass_next` into a pure forward that, given the clamped inputs and the
  current free states, returns every quantity F needs (the 5 image-side taps, the 5 text-side taps,
  the two reconstructions). It must be tape-friendly, no in-place Variable writes inside the forward.
- REWRITE `train_step` and `test_step` around the two-tape scheme in section 3.

---

## 2. The single energy F for the big model

Free states. The 5 shared latents `S1..S5` (the `share_state` Variables). Clamped. The image input
`X_img` and the text input `X_txt` (during joint training both are clamped). Weights are constants in
the relaxation tape and the variables of interest in the learning tape.

Edges, all squared prediction errors, mean over batch.

- Cross-modal coupling (encoder edges, precision `a_cross`). For each scale k in 1..5, the image
  encoder produces a tap `a_img_k` (forward conv tower up to that scale, then the existing
  `flatten -> inter -> dense` projection into the shared space) and the text encoder produces
  `a_txt_k` (forward transformer pyramid up to that scale, then its projection). Each shared latent
  carries two errors, `||S_k - a_img_k||^2 + ||S_k - a_txt_k||^2`. Both modalities predict the shared
  latent. This is the L3 per-scale anchoring, one anchored shared latent at each of conv9/8/6/4/2 and
  the matching transformer depths.
- Generative reconstruction (decoder edges, precision `a_gen`, L4). Image reconstruction
  `||X_img - dec_img(S)||^2` and text reconstruction `||X_txt - dec_txt(S)||^2`, where `dec_img` and
  `dec_txt` are the existing dense head stacks that map the shared latents back to pixel and token
  space. The clamped real inputs are the targets, which is the anti-collapse anchor (L1).

F = 0.5 * mean ( a_cross * sum_k ( image-encoder error_k + text-encoder error_k )
              + a_gen   * ( image reconstruction + text reconstruction ) )

Precision (L4). Start `a_gen = 2`, `a_cross = 1`. The miniature needed `gen >= disc` or text->image
collapsed to chance through a coupling shortcut. The same risk exists here because the caption and
the image are perfectly paired. Treat the ratio as the first thing to sweep on the pod if generation
is weak. Per-scale precision may also be needed if deep scales lag (Stage 1.5).

Note on the existing topology. The 5 shared states are LARGE reconstruction-space vectors (102400 up
to 1429912), so in this model the shared latent and the reconstruction live close together. That is
fine for F as written. It does mean the coupling and the reconstruction are not as cleanly separated
as in the toy, which is one reason to watch the `a_gen` vs `a_cross` balance closely.

---

## 3. The tape structure (L2, the silent-failure risk)

Two distinct phases per train step, both differentiating the SAME F.

Phase A, relaxation (inference). Goal, move the free states toward equilibrium with weights fixed.
- Tape watches ONLY the 5 shared-latent state tensors. Weights are read as constants.
- Inside the tape, compute F from the clamped inputs and the current states. Because weights are
  fixed, the encoder taps and the decoder outputs that depend only on clamped inputs and weights are
  constant across relaxation steps and can be precomputed once per step (this is the legitimate
  efficiency move, the same one the inference functions in the stage files use).
- `g = tape.gradient(F, states)`, then `state <- state - beta * g` for N_infer steps.
- Correctness, this tape must NOT update weights. The encoder being outside this tape is correct here
  because inference holds weights fixed.

Phase B, learning. Goal, one weight step from the relaxed states.
- States are now constants (the relaxed values from Phase A).
- The tape watches the trainable weights. CRITICAL, inside this tape recompute the FULL forward from
  the clamped inputs, the entire conv tower, the entire 17-block transformer pyramid, and BOTH
  decoder stacks, then F. If any module is computed before the `with tape` block or read from a cache,
  it gets a None gradient and freezes silently. This is exactly the Stage 1.6 bug, where the
  transformer sat outside the tape and never trained while accuracy still looked plausible.
- `grads = tape.gradient(F, model.trainable_variables)`, then `w <- w - alpha * grad`. Plain SGD, no
  LARS, no clips, no Adam, to keep F observable, matching the stages.

Mandatory guard in the smoke test. Print the gradient norm for at least one weight in the conv tower,
one in the transformer pyramid, and one in each decoder, and assert all are finite and nonzero. A
None-to-zero guard like the one that hid the Stage 1.6 bug must be removed or must raise, not
silently zero-fill.

---

## 4. Conv tower and transformer as feed-forward autodiff edges (not per-layer free states)

Decision. Treat the conv tower and the transformer pyramid as feed-forward autodiff edges. Do NOT
give each conv layer and each transformer sublayer its own free state with its own relaxation. The
only free states are the 5 shared latents.

Why this is safe. Stage 1.5 measured that relaxing the per-layer conv states changed the readout by
nothing, 49.2 percent free versus 49.2 percent pinned, while the per-layer weights still trained with
balanced gradients. Stage 1.6 then built both encoders as feed-forward edges and still generated
recognizably (8 of 10 digits) and reconstructed faithfully (recon MSE 0.032), so the feed-forward
choice holds for the generative direction too, not just discrimination.

What it removes.
- The per-layer STATE Variables for conv1..9 and for every transformer sublayer. These were stored as
  `tf.Variable`s in the current design and are persistent memory. The `conv2` state alone is about
  568x568x64 per batch element. Removing all of them frees a large block of persistent memory.
- The inference iteration over those states. The current `update_states_wts_b` loops over the full
  `trainable_layers` list `num_steps` times. After the change, relaxation iterates over only the 5
  shared latents, so the per-step inference cost drops from "all layers times N_infer" to "5 states
  times N_infer", a large reduction in both compute and the number of Variable writes.

What stays free. The 5 shared-latent states, plus, only at test time, the input state of whichever
modality is being predicted (clamp text, free image, for text->image, and vice versa).

Caveat to verify on the pod. The feed-forward equivalence was shown on a 3 to 5 layer tower. The big
tower is 9 conv layers plus a 17-block transformer pyramid. The expectation is that feed-forward is
still fine because gradients flow by autodiff, but the per-scale grad-health check (section 7b) is the
gate that confirms no scale is starved.

---

## 5. The giant reconstruction heads (about 6B of 7.7B params, and they are the decoders)

The `flatten -> Dense(100) -> Dense(huge)` heads are simultaneously the parameter monsters and the
generative decoders that L4 requires.

DECISION. Option K, KEEP the flatten-dense reconstruction heads. This was previously a recommendation
with a conv-decoder replacement (Option R) held as an opt-in. Stage 1.7
(`stage1/stage1_7_conv_decoder_pcn.py`) ran that experiment as a miniature and the result reverses
the temptation to replace them. Option R is now ruled out for this model.

What Stage 1.7 did. A controlled same-seed head-to-head off the passing Stage 1.6 model, changing ONE
thing only, the image decoder. The flatten-dense decoder versus a conv/deconv decoder
(`dense-project to 7x7x16 -> nearest-upsample + conv, twice -> sigmoid`). Everything else identical.

| metric | conv decoder | dense (flatten) |
|---|---|---|
| image-decoder params | 152,545 (61 percent) | 250,896 |
| image to text accuracy | 0.640 | 0.626 |
| image to image recon MSE | 0.012 (sharper) | 0.032 |
| text to image per-class | 0.40 (collapses) | 0.90 |
| attention grad, generative dir | 0.27 | 0.17 |
| multi-scale anchor health | balanced, spread 7x, none frozen | same structure |

The deciding fact. The conv decoder has fewer params and reconstructs images BETTER (image to image
MSE 0.012 versus 0.032, visibly crisp), yet text to image generation COLLAPSES, per-class re-read
0.40 versus the dense head's 0.90, visually fragments and a mode-collapse to a generic stroke for most
classes while the dense head emits recognizable digit prototypes. The text-clamped relaxation energy
first rose, which looked like instability, but a smaller generative step fixed the descent and
generation still collapsed, so the collapse is a real generalization limit, not a step-size artifact.

The interpretation. A sharper conv decoder overfits the in-distribution, image-driven latent, which is
exactly why its reconstruction is crisp, but it generalizes the off-manifold TEXT-only latent worse,
so cross-modal generation collapses. The smoother dense head tolerates weak text conditioning and
still emits a recognizable prototype. Better reconstruction traded against generation. Generation IS
the purpose of this model (bidirectional image to text and text to image), so the dense head wins.

L3 is not the concern here. The multi-scale per-scale anchors survived the decoder swap intact
(per-scale state movement about 0.4 to 0.7, gradient spread 7x, no frozen scale). The decoder swap
did not re-break depth. The failure was purely text-side generation quality.

Consequence. Keep the heads, the `inter9` matrix stays about 2B params and static weights stay around
28.7 GiB, and the memory cost of the autodiff backward graph is paid by gradient checkpointing in
section 6, not by replacing the decoders.

---

## 6. Revised memory and pod profile

Because section 5 keeps the about 6B-param heads (Option K), the heads stay in memory and gradient
checkpointing becomes the PRIMARY memory lever, not an optional one.

- Static weights. About 28.7 GiB, unchanged, since the heads are kept. The conv-decoder route that
  would have cut this is ruled out per section 5.
- Removed by section 4. The per-layer conv and transformer STATE Variables, persistent memory, freed.
- Added by L1, the honest flag. Autodiff of one F over the whole graph materializes a backward graph
  storing forward activations for the backward pass. The current hand-written scheme never built that
  graph, so peak ACTIVATION memory under autodiff can be higher than the old run, especially across
  the 2B-param `inter9` head and the 17-block pyramid. This is a real new cost not present before, and
  it may push peak above the old 60 to 67 GiB.
- Primary lever. Gradient checkpointing (`tf.recompute_grad`) on the giant heads and on the
  transformer blocks, trading compute for activation memory. This is now load-bearing, not optional,
  because the heads are staying. It changes nothing about the model or the math.
- Always-on supporting levers, not architecture changes.
  - `TF_GPU_ALLOCATOR=cuda_malloc_async`, which already took the old run from OOM to a completed step.
  - `jit_compile=True` on both step functions, XLA fusion, proven safe and faster in the stages.
- Fallback order if checkpointing plus an 80 GB pod still does not fit. A bigger GPU (for example
  H100 80 GB or a multi-GPU split), then a smaller batch, then activation or optimizer-state offload to
  host memory. NOT swapping the decoders. Stage 1.7 showed conv decoders break text to image, which is
  the model's core capability, so decoder replacement is off the table as a memory fix.
- Target to confirm. An 80 GB pod, `TF_GPU_ALLOCATOR=cuda_malloc_async`, `jit_compile=True`, with
  gradient checkpointing enabled from the start.

---

## 7. Staged pod validation (each stage gated, do not proceed on failure)

This mirrors the staged miniature ladder. It is validation, not "run training".

- 7a. Smoke test. Rewritten `train_step`, batch 1, one forward plus a few weight steps.
  - Assert F is finite and DESCENDS during relaxation in BOTH clamp directions, clamp-image and
    clamp-text, like Stage 1.6 A.
  - Assert states stay finite.
  - L2 guard. Print per-module gradient norms, conv tower, transformer pyramid, image decoder, text
    decoder, and assert every one is finite and nonzero. This is the check that would have caught the
    Stage 1.6 freeze.
- 7b. Per-scale grad health (Stage 1.5). Over a handful of steps, log the gradient norm for a weight
  at each of the 5 scales and across conv depth. Assert no scale is frozen and the spread is not
  orders of magnitude (the failure was a 400x spread when only the top was anchored). If a deep scale
  lags, add a per-scale precision bump or an extra anchor before proceeding.
- 7c. Bidirectional check (Stage 1.6) on a small real image-text subset, for example a few hundred
  COCO pairs. Image->text must score above chance on a simple caption metric, and text->image must
  produce output recognizable as the conditioned content (eyeball a grid, and optionally score with an
  off-the-shelf classifier or CLIP similarity). Confirm `a_gen >= a_cross` is actually needed here by
  trying the inverted ratio once and showing generation degrades, to verify L4 transfers.
- 7d. Only after 7a to 7c pass, a longer training run. Watch F trend, the per-scale grad health, and
  periodic image->text and text->image samples.

---

## 8. Risks and open questions, honestly marked

Solid, validated in the miniatures and expected to transfer.
- One energy plus autodiff removes the inconsistent hand-derivatives and the energy descends (Stages
  0, 1, 1.5, 1.6).
- The tape-scope rule and the explicit per-module grad guard (L2). Solid, and the guard makes the
  failure loud instead of silent.
- Dense per-scale anchoring prevents the deep-layer freeze (L3). Solid on 5 layers, the big tower is
  the scale-up test.
- Feed-forward encoders lose nothing on the readout and still generate (Stage 1.5 and 1.6). Solid in
  the toy, 7b is the confirmation at depth.

Bet, not yet validated, watch closely.
- The transformer PYRAMID with sequence-resizing across 5 scales. The miniature used ONE transformer
  block at ONE scale. The interaction of `linear_2=48 / linear_4=12 / linear_6=3` sequence resizing
  with per-scale anchoring is untested. This is the biggest unknown.
- Backprop-like attention at scale. Per Rosenbaum 2021 (arXiv 2106.13082) autodiff through attention
  is backprop-like, not strictly free-energy-following. Across 17 blocks this could behave differently
  from a single block. Stability and the energy-descent guarantee through the deep pyramid are a bet.
- Real data versus clean tokens. The miniatures used MNIST pixels and clean digit-token captions. Raw
  572x572x3 COCO images and real word or character captions are harder, and the `a_gen >= a_cross`
  balance that worked on the toy may need re-tuning. 7c tests this directly.
- Autodiff peak memory over the 2B-param heads. May exceed 80 GB even with `cuda_malloc_async` and
  jit. Gradient checkpointing is now the primary mitigation (section 6), but whether it is sufficient
  with the heads kept is unverified. If it is not, the fallback is a bigger GPU, smaller batch, or
  activation offload, NOT swapping decoders, since Stage 1.7 showed conv decoders break text to image.
- The 5 shared latents are large reconstruction vectors rather than small bottleneck codes, so the
  coupling and the reconstruction are entangled. Whether the `a_gen` vs `a_cross` split behaves as
  cleanly as in the toy, where the latent was small, is a bet.

---

## Decisions requested before any code

1. The giant heads. DECIDED, Option K, keep the flatten-dense heads, per the Stage 1.7 evidence in
   section 5. Conv-decoder replacement (Option R) is ruled out because it collapses text to image.
2. The starting precision ratio `a_gen = 2, a_cross = 1`, and agreement to sweep it on the pod.
3. Confirm an 80 GB pod and that gradient checkpointing is acceptable as the memory lever.
4. Confirm the staged gating in section 7, in particular that 7a to 7c must pass before 7d.
