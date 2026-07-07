# Design: PC-native InfoNCE coupling for the bidirectional class

Date: 2026-07-07

## Goal

Fix the text->image failure of the bidirectional class (SP2 quick-look: text->image
generates uniform blobs because the shared latent is image-dominated) by adding an
InfoNCE contrastive coupling that pulls the image-branch and text-branch codes together
for matched pairs and apart for mismatched ones, so the caption produces a
representation that can drive image generation. The coupling must be PC-native: no
backprop through the network. Weight learning stays 100% predictive coding.

## Hard constraints (inherited + this feature)

- The bidirectional class only; learning stays predictive coding. The InfoNCE gradient
  is computed ONLY with respect to the codes (closed form over the batch) and injected
  as an extra error at the code layer during relaxation; the WEIGHT updates remain the
  existing local beta-less LARS. NO backprop through the network. NO Adam/momentum.
- The five shared-latent pairs stay aliased. NATIVE_7B unchanged: InfoNCE is opt-in
  (off by default), so NATIVE still GATE_MATCHes.
- Runs on the H200 via tools/clusterrun.sh; commits first-person student, no AI attribution.

## Background (mechanism, verified)

The deepest shared scale is image `dense2` (shared state) fed by the 100-dim bottleneck
`inter2`, and text `dense4` (aliased to `dense2`) fed by the 100-dim bottleneck
`inter12`. So `inter2.state` is the image-branch code and `inter12.state` is the
text-branch code, both 100-dim, both present in a single both-clamped relaxation
(approach A). The dense/inter `update_state` relaxes via
`state.assign_sub(state_lr * errors)`, so an extra InfoNCE error term is added there.

## Component 1: the codes to contrast

During the normal both-clamped relaxation, per batch:
- `u = inter2.state`  (image-branch code, shape (B,100))
- `v = inter12.state` (text-branch code, shape (B,100))
L2-normalize each row. This is approach A (chosen): one relaxation, standard two-branch
CLIP-style codes, no extra conditional relaxations. Applied at the DEEPEST shared scale
first (`inter2`/`inter12`); extensible to the other four scales later.

The constructor stores references to these two layers (e.g. `self._infonce_codes =
(inter2, inter12)`) so the training loop can read/inject without hardcoding indices.
Storing the references is inert when InfoNCE is off.

## Component 2: the InfoNCE gradient (closed form, no backprop)

Symmetric InfoNCE over the batch with temperature tau (default 0.07):
- logits `S = (u_norm @ v_norm^T) / tau`  (B,B); targets = identity (matched pairs on the diagonal).
- `L = 0.5*(cross_entropy(S, I) + cross_entropy(S^T, I))`.
- The gradients `dL/du` and `dL/dv` are the standard softmax-contrastive expressions,
  computed directly from `u_norm`, `v_norm`, and the softmax of `S` — a closed-form
  batch operation on the codes, NOT a backprop through the encoders. (Implementation may
  use a `tf.GradientTape` scoped ONLY over the `u,v -> L` code-to-loss computation to get
  `dL/du, dL/dv`; the tape must NOT extend into the network weights. This is a convenience
  for the closed form, not network backprop.)

## Component 3: injection into relaxation + PC weight learning

During the relax sweep, after `inter2`/`inter12` take their normal `update_state`, add the
InfoNCE error to each: `inter2.state.assign_sub(state_lr * lambda * dL/du)` and likewise
`inter12` with `dL/dv`. The shared state then relaxes to reduce reconstruction AND
InfoNCE; the existing local LARS weight step (unchanged) learns weights that produce
aligned codes. `lambda` weights the coupling (tuned). This is the only new term; the
per-layer weight update math is untouched.

Training is JOINT (chosen): reconstruction + lambda*InfoNCE together from the start (no
separate warm-up). Config is the best-so-far: lr=1e-3, weight_decay=3e-2,
state_clip=400, gelu conv, relaxed schedule (relax 15, batch 8), plus `--infonce-lambda`
and `--infonce-tau` args (lambda=0 -> InfoNCE off -> current behavior).

## Component 4: validation

- NATIVE_7B GATE_MATCH (InfoNCE off by default -> byte-identical; the stored references
  and the lambda=0 path change nothing).
- COCO64 with InfoNCE on: train the 2k overfit and check the real question — does
  text->image move OFF the uniform blob toward structure (PNG size / visible content),
  and does the image-code vs text-code alignment improve (batch retrieval accuracy of
  u<->v, or mean matched-pair cosine rising vs mismatched). Image->caption and
  reconstruction must stay intact. Energy should still descend (coupling should not blow
  up training; watch max|state|).

## Risks and unknowns

- InfoNCE may not fix text->image even coupled (the functional version could not crack
  cross-modal generalization with InfoNCE + warmup + scale). In-sample text->image
  improvement is the first, achievable check; held-out is later and may still fail.
- Injecting a batch-global error into the per-example relaxation is novel; it may
  destabilize the relaxation (watch energy / max|state|; lambda and tau are the levers).
- The 100-dim bottleneck may be too small to carry alignment; if so, contrast a
  different/added code layer (a follow-up, out of scope here).

## Out of scope

Multi-scale InfoNCE (only the deepest scale here), a separate InfoNCE warm-up schedule,
approach B (conditional-relaxation codes), held-out evaluation, and any backprop-based
InfoNCE. Each is a later step if the in-sample text->image signal warrants it.
