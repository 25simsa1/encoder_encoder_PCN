# Design: Make the bidirectional PCN actually generate (both pathways)

Date: 2026-07-06

## Goal

Get `EncoderEncoderPCN` (the real bidirectional predictive-coding class in
`encoder_encoder_pcn.py`) to train on COCO and generate in both directions:
clamp the caption and relax to produce an image, and clamp the image and relax to
produce a caption. Success milestone one is recognizable IN-SAMPLE generation both
ways; held-out comes after.

## Hard constraints

- The model is the bidirectional class, always. One conv image network used both
  directions via `predict_prev` (encode) / `predict_next` (decode) with shared
  weights; one transformer text network the same; joined by shared latent states
  (`share_state_layer` ties `dense4<->dense2`, `dense8<->dense6`, `dense12<->dense10`,
  `dense16<->dense14`, `dense20<->dense18`). Trained relax-then-step
  (`update_states_wts_b_relaxed`). Generation is `test_step(predict='img'|'txt')`.
- Learning stays predictive coding (relaxation + local weight updates). No backprop
  diffusion, no off-the-shelf model, no pretrained components. This keeps the model
  from-scratch, which is the point of the work.
- The functional scale scripts (`run_coupling_scale.py`, `run_coupling_capacity.py`,
  etc.) are retired for this work. Everything runs on the class.
- Deadline is not a constraint on the design; correctness and staying bidirectional
  are.

## Starting point (measured, job 8299)

Native config is 572x572x3 images, 192x512 text, 7.702B params (30.8 GiB weights,
dominated by the flatten->dense projection layers). At batch 1: pass_through peak
34.3 GiB; ~14 s per relax+weight sweep (eager, per-step `gc.collect()`); memory
grows 34->68 GiB over 3 steps (accumulation); states 0.37 GiB at batch 1, ~47 GiB
at batch 128; stable (no NaN) for 3 steps. As-is it is not trainable: too slow,
memory-leaking, batch-1 only, and the wrong resolution.

## Phase 2: execution rewrite (prerequisite; everything depends on it)

Make the class trainable without changing the PC math.

- Remove the per-step `gc.collect()` from `update_states_wts_b` /
  `update_states_wts_b_relaxed`.
- Wrap the relaxation sweep and the weight step in `tf.function` graph mode so the
  hundreds of per-layer Python calls compile to a static graph instead of eager
  iteration. Expect the ~14 s/step to drop by one to two orders of magnitude.
- Diagnose and fix the 34->68 GiB growth over 3 steps (likely eager op/tensor
  retention across steps or per-call graph rebuild in `pass_through`).
- Enable real batching (batch > 1) and confirm state/weight shapes carry the batch
  dim correctly.
- VALIDATION GATE: on a tiny configuration, the rewritten loop must reproduce the
  current class's states and outputs within numerical tolerance for a few steps,
  proving the rewrite changed speed/memory only, not the algorithm.

Done when: a batched relax+weight step runs in graph mode, memory is flat across
many steps, and the tiny-config outputs match the pre-rewrite class.

## Phase 3: retarget to 64px

Resize the architecture from 572px to 64px COCO while keeping the bidirectional
structure and the 5-scale shared-latent wiring intact.

- Image input becomes 64x64x3. The conv stack (`conv1`-`conv9`, four maxpools)
  produces much smaller feature maps at 64px, so the flatten sizes and therefore the
  dense projection input dims change; re-pick the projection widths (the current
  2B/1B/0.5B dense layers are sized for 572px) down to sane values for 64px.
- Keep the multi-scale taps (at `conv2/4/6/8/9` and the matching transformer depths)
  and the five shared-state dense pairs.
- Keep the transformer text pathway; set the caption sequence length to the COCO
  caption format we feed.
- Target a size that trains and batches on one H200 (order hundreds of millions of
  params, not billions).

Done when: a 64px bidirectional config instantiates, trains a few steps batched
within H200 memory, and preserves the shared-latent bidirectional wiring.

## Phase 4: train on COCO-64 and evaluate both pathways

- Adapt a data feeder to the class's input format (image `(B,64,64,3)`, text
  `(B,seq,512)` plus mask), reusing the existing COCO train2017 cache.
- Train with relax-then-step, small first (a few thousand pairs), then scale.
- Evaluate both directions with `test_step`: `predict='img'` (caption -> image) and
  `predict='txt'` (image -> caption). In-sample recognizability first, then held-out.
- Reuse the existing eval discipline (held-out latent retrieval, a fixed qualitative
  panel, generation metrics), reporting both pathways.

Done when: for training captions, the model produces recognizable, caption-varying
64px images and plausible captions from images. Then push to held-out.

## Risks and unknowns

- Graph-tracing hundreds of PC layer objects may trace slowly or retrace; may need
  `reduce_retracing`, stable shapes, or restructuring the loop.
- The 34->68 GiB growth may be structural, not a one-line fix.
- Retargeting the dense widths changes the model; PC stability at the new size is
  unknown (the LR/divergence lessons from the functional runs may or may not
  transfer).
- The class's data-input format differs from the functional pipeline; the feeder is
  new work.

## Out of scope

Deadline-driven shortcuts; the functional version; any non-PC training; pretrained
components; resolutions above 64px (later, if 64px works).
