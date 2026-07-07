# Design: Phase 4 sub-project 1 — COCO64 data feeder + first training run

Date: 2026-07-07

## Context and decomposition

Phase 4 (train the bidirectional class on COCO-64 and evaluate both pathways) is
decomposed into two sub-projects, each with its own spec/plan cycle:

- SP1 (this spec): the COCO64 data feeder plus a first training run that overfits
  ~2k pairs. Deliverable is a checkpoint that trains sanely (energy behaving, no
  divergence) plus a reusable feeder.
- SP2 (later): evaluation and generation quality (both directions, in-sample first,
  then held-out).

The milestone this serves (from the parent spec) is in-sample recognizability first.
SP1 gets the machinery training; SP2 judges whether it generates.

## Goal

Train `EncoderEncoderPCN(config=COCO64_156M)` on ~2000 COCO image-caption pairs for
many epochs, using the class's own relaxed predictive-coding schedule, and produce a
loadable checkpoint with an energy log showing the model is fitting the set and not
diverging.

## Hard constraints (inherited)

- The bidirectional class only (`encoder_encoder_pcn.py` + `*_pcn_layer.py`). PC
  learning only, no functional-version model code, no backprop, no pretrained
  components (the character vocabulary and embedding are learned from scratch).
- The five shared-latent pairs stay aliased (`share_state_layer`).
- Runs on the Colby H200 via `tools/clusterrun.sh` with the existing `~/tf-env`.
- Commits: first-person student voice, no AI attribution.

## The optimizer is the class's, not an add-on (verified)

The per-layer weight update in `dense_pcn_layer.py:128-131` and
`conv_pcn_layer.py:130-133` is a beta-less LARS trust-ratio step,
`w -= learning_rate · (‖w‖/(‖g‖+1e-6)) · g`. There is no weight decay, no momentum,
and no Adam anywhere (verified by grep across the three files). The trust ratio scales
WEIGHTS ONLY. The bias (dense only) uses a plain `bias_lr` step with no trust ratio
(`dense_pcn_layer.py:160`); the state relaxation uses a plain `state_lr` step with no
trust ratio (`dense_pcn_layer.py:81,94`, `conv_pcn_layer.py:83,101`). SP1 adds no
optimizer machinery; it only sets the learning rate and runs the schedule.

**state_lr and the weight learning rate are separately tunable.** `state_lr`,
`bias_lr`, and `learning_rate` are separate layer attributes, all initialized to the
single value passed at construction. So one number (2e-2) sets the trust-scaled weight
rate, the plain bias rate, and the plain relaxation rate together. SP1 starts them
coupled at 2e-2. If the latents relax too aggressively during the overfit run (energy
spiking from the relaxation, not the weight step), the relaxation rate can be lowered
independently of the weight LR by setting `state_lr` on the layers after construction.
Keep them coupled unless the energy misbehaves.

## Component 1: data feeder (`coco64_data.py`)

A standalone, reusable module (SP2's evaluation imports it too).

- Images: load `~/coco_scale/imgs_sc_train2017.npy` `(22000,64,64,3)` float32 in
  [0,1], take a fixed first-2000 subset, feed as the `(B,64,64,3)` image input
  unchanged (no preprocessing; it is already the COCO64 image format).
- Captions: load `~/coco_scale/caps_sc_train2017.txt` (one per image, aligned by line
  order to the image array). Build a FIXED character vocabulary from the training
  captions plus `<pad>` and `<unk>`, freeze it to a file (`coco64_char_vocab.json`) so
  SP2 uses the identical mapping, and record its size V.
- Encode each caption to a one-hot character sequence of length 64 (truncate longer,
  right-pad shorter with `<pad>`, map unseen chars to `<unk>`), giving a `(B,64,V)`
  text input.
- Mask: a `(B,64)` additive attention mask, 0 at real character positions and a large
  negative value at padding, matching what the transformer's `attention += mask`
  expects (the reduction of this mask through the text bridges is already handled in
  `pass_next`, keyed on the config bridge seq-lens).
- Expose `decode(indices) -> text` for SP2.
- Batching: yield `(img (B,64,64,3), txt (B,64,V), mask (B,64))` over the 2k subset,
  shuffled per epoch.

## Component 2: config update (`pcn_config.py`, COCO64_156M only)

Character-level text changes two config fields: `txt_seq_len` 32 → 64, and
`txt_embed_dim` 512 → V (the frozen char-vocab size), since the text input is now
one-hot characters and the model's `txt_embedding` learns the V→512 projection. This
shifts the parameter count slightly, so re-verify COCO64 builds and lands in the
125M-190M band, re-tuning widths only if it drifts out. While editing the config, fold
in the two cheap Phase-3 deferrals: add `PCNConfig.__post_init__` length asserts (so a
malformed capacity-axis config fails loudly), and re-confirm the mask reduction still
fires for the new text config (COCO64 bridge seq-lens 16/8/4 stay disjoint from every
other Dense width, so the existing value-keyed condition is correct here; full
role-keying stays deferred to when the capacity-axis configs are defined).

## Component 3: training script (`train_coco64.py`)

- Build `EncoderEncoderPCN(2e-2, config=COCO64_156M)`.
- Per batch: clamp both inputs (`img_input.is_clamped = txt_input.is_clamped = True`),
  `pass_through(img, txt, mask)`, then `update_states_wts_b_relaxed(num_weight_steps=1,
  num_relax_steps=R)` with R around 10-20 (relax the latents toward equilibrium, then
  one LARS weight step). Loop over the 2k subset for many epochs.
- Learning rate 2e-2 (the scale where the functional version was stable), plain
  beta-less LARS as above. No weight decay (we are overfitting), no LR ramp, no
  InfoNCE warmup (those were 3B/7.7B or coupling-experiment aids, not this run).
- Batch size 8-16 (memory is ample at 64px/158M).
- Checkpoint the model weights every ~1000 steps and at the end; support resume.
- Log the PC energy (total prediction error) each step and a non-finite check, so a
  divergence is caught early.
- Divergence levers, in order, if the energy blows up (the 7.7B failure mode): lower
  the weight LR, add a short LR ramp, lower `state_lr` independently, then enable
  `state_clip`. Log which lever was used.

## Validation and success criteria

- Feeder: local unit test (batch shapes `(B,64,64,3)`/`(B,64,V)`/`(B,64)`, images in
  [0,1], a caption round-trips encode→decode within truncation/`<unk>`, mask is 0 at
  real positions and negative at pad, vocab file freezes and reloads).
- Config: COCO64 rebuilds and the param count is in [125M,190M] after the seq/embed
  change; the config asserts pass; `NATIVE_7B` is untouched.
- Training: a short smoke first (a few steps over a couple of batches on the H200,
  energy finite and moving, checkpoint saves and reloads), then the real 2k overfit
  run.
- SP1 is done when the 2k overfit run completes on the H200 without divergence, the
  energy trend is down or stable (the model is fitting the set), and a loadable
  checkpoint exists.

## Risks and unknowns

- PC stability at 64px/158M is unproven at training length. SP1's monitor + levers
  address this; a persistent divergence would be a real finding (report, do not force).
- Whether the model actually fits/memorizes 2k pairs is the open question SP1 answers
  at the energy level; recognizable generation is SP2.
- The char-vocab must be frozen so SP2 decodes with the identical mapping.

## Out of scope (SP2 and later)

Evaluation, generation quality, both-direction generation panels, retrieval/held-out
metrics, scaling past 2k pairs, and the mask role-keying (deferred until the
capacity-axis configs are defined). Each is later work.
