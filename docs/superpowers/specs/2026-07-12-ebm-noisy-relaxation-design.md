# Design: PC-native EBM by noisy relaxation (Approach B1)

Date: 2026-07-12

## Goal

Break the mean-collapse for text-to-image by treating the bidirectional PC net as an energy-
based model: sharpen the energy landscape with a contrastive-divergence weight rule and GENERATE
BY SAMPLING (noisy relaxation) instead of reading the blurry mode. Strictly within bidirectional
predictive coding, on the 2k COCO64 overfit. This is the escalation after Approach A (high-
frequency-weighted energy) came back a fair null.

## Background (why the mode is blurry, why sampling helps)

The PC relaxation is gradient descent on ONE scalar energy (the sum of local prediction errors,
both `predict_next` up and `predict_prev` down). In generation we clamp the caption and relax the
free image and latents, and the image settles at the energy's mode conditioned on the text-set
latent, which is the blurry conditional mean. Approach A confirmed that reweighting the
deterministic error cannot beat that mean. The principled cure is to (1) shape the energy so real
images sit in sharp low-energy wells (contrastive divergence, which the CHL contrast already
approximates), and (2) SAMPLE the image from that energy by noisy (Langevin) relaxation, so the
output is a specific sharp image rather than the average. PC already IS an EBM; we were just
reading its mode instead of sampling it.

## This is bidirectional PC

Everything is the class's own `update_state` (plus a noise draw) and its own local `update_wts`
(the sign-flipped contrast). One shared-weight net used both directions, no backprop, no separate
decoder, no optimizer. Off (temperature 0, no EBM step) is byte-identical.

## Component 1: noisy (Langevin) relaxation in `update_state`

Add an opt-in per-layer `noise_temp` (float, default 0.0) to the state-bearing layers
(`Conv2DPCNLayer`, `DensePCNLayer`, `InputPCNLayer`). At the END of `update_state`, after the
deterministic assign_sub(s) and BEFORE the `state_clip`, add a Langevin noise draw when on:

  if self.noise_temp > 0.0:
      self.state.assign_add(tf.sqrt(2.0 * self.state_lr * self.noise_temp) * tf.random.normal(self.state.shape))

This turns the relaxation step `state <- state - state_lr*grad` into
`state <- state - state_lr*grad + sqrt(2*state_lr*T)*xi`, i.e. Langevin dynamics that sample the
energy at temperature `noise_temp`. `noise_temp == 0` skips the branch, so the layer is byte-
identical. `noise_temp` is a plain Python float (so the compiled sweep's branch resolves at trace
time). It is set/annealed by the caller (the EBM step and the generation loop), NOT a constant on
the layer.

## Component 2: contrastive-divergence EBM training step

A new `ebm_step` (module-level in `train_coco64.py`, alongside `chl_step`), and `--train-mode ebm`.
It has the SAME structure as the CHL step, with the negative phase made stochastic:

- Phase 0 (condition on the caption): caption clamped, image = zeros unclamped, relax `k0` so the
  shared latents become text-set; then HOLD the latents fixed (unclamped, excluded from the loops)
  for both phases, so the contrast varies only the image (identical to `chl_step`).
- NEGATIVE phase (model sample, CD-1 from data): set `img_input` to the TRUE image, free it, and
  noisy-relax the decode `k1` with ANNEALED `noise_temp` (T0 -> ~0 over the k1 steps) to draw a
  sample near the data, then RAISE its energy with the local weight step at -gen_lr. This IS
  `chl_step`'s free/anti-learn phase, now stochastic (and started from the data instead of the
  passthrough, which is what makes it CD-1).
- POSITIVE phase (data): clamp `img_input` to the TRUE image, relax the decode `k2` (deterministic,
  latents fixed), then LOWER its energy with the local weight step at +gen_lr. This is `chl_step`'s
  clamped/learn phase, unchanged.
- Net local update: `wts -= gen_lr*(g_positive_data - g_negative_sample)` per decode layer, exactly
  the CHL sign-flip machinery, but the negative phase is a NOISY CD-1 sample rather than the
  deterministic mean. Weight decay off during both, ends in the recon clamp config.

The only change from `chl_step` is that the free/negative phase sets an annealed `noise_temp` on
the free image (and optionally the decode states) so it draws a sample instead of relaxing to the
mean. Everything else (clamp hygiene, latent-by-exclusion, the ±gen_lr sign flip, weight decay off)
is identical and already reviewed.

## Component 3: generation by sampling

Generation clamps the caption and NOISY-relaxes the free image and latents with an annealed
`noise_temp`, then reads `img_input.predict_next()` as ONE sample. The retest path (`darkness_diag`
regime T, and the top-down-boosted generation) gains a `--noise-temp` / anneal so it samples the
CD-trained energy instead of relaxing to the mode. Draw a few samples per caption to show the
generation is stochastic and sharp, not a single blurry mean.

## Component 4: opt-in and flags

- `--train-mode ebm` selects the CD step (recon and CHL modes unchanged).
- `--noise-temp T0` (float, default 0.0) the initial Langevin temperature; `--noise-anneal` the
  decay (e.g. linear T0->0 over the negative-phase / generation relax steps). Off (T0=0) is
  deterministic and byte-identical.
- `noise_temp` defaults 0.0 on every layer; NATIVE and every non-ebm run are byte-identical, so the
  golden gate (`GATE_MATCH nlayers=143`) and the COCO64 gate hold. Composes with `--weight-norm`
  (keep it on for stability) and the existing `--state-clip`.

## Validation

- NATIVE `GATE_MATCH nlayers=143` (and the COCO64 gate) with noise off and no ebm step: byte-
  identical, since the `noise_temp==0` branch is skipped. The canonical NATIVE-143 gate stays
  deferred to a big-GPU window (H200 drained); the COCO64 gate + provable inertness are the interim
  proof.
- A local unit test: `noise_temp=0` leaves `update_state` byte-identical; `noise_temp>0` adds a
  zero-mean perturbation of the expected scale and only on the intended layers.
- A short stability smoke on COCO64_GEN (`--train-mode ebm --weight-norm --noise-temp <T0>`): does
  CD training stay finite and bounded (CD + noise can destabilize; pair with weight-norm and
  state-clip)?
- The decisive retest on the 2k overfit: CD-train warm-started from the recon best, then GENERATE BY
  SAMPLING (annealed noisy relaxation) and run `darkness_diag` plus save a few sampled PNGs per
  caption. Does the sampled-generation contrast rise off the 0.25 plateau, is brightness off ~0.40,
  and do the SAMPLES look sharp and caption-specific (and vary across draws)? Sweep T0 and the
  anneal to find a working regime.

## Risks and unknowns

- Deep contrastive-divergence EBM training is finicky (poor mixing, mode collapse, instability). The
  weight-norm stabilizer, state-clip, CD-1-from-data (a short, near-data negative chain), and the
  temperature anneal are the guards; the T0/anneal sweep is the tuning.
- If the energy does not sharpen, the samples will be noisy blur rather than crisp images; that is
  the make-or-break, and it points to the diffusion flavor (multi-noise-level denoising) as the next
  escalation if B1 plateaus.
- Adding noise to `update_state` touches gate-critical layer code; the `noise_temp==0` path must stay
  byte-identical (the gate is the guard).
- Sampling adds relaxation cost and stochasticity to generation; the retest shows a few draws per
  caption rather than one.

## Sub-decisions (chosen defaults)

- Negative sample = CD-1 from the data (start the negative chain at the clamped image, take `k1`
  noisy steps). Simplest stable EBM negative; persistent chains and from-noise are fallbacks.
- Noise schedule = annealed temperature (T0 high to ~0 over the relax steps), so the sample explores
  then settles into a sharp well, which suits the overfit where each caption has one target image.

## Out of scope

The diffusion flavor (B3, multi-noise-level denoising) and the latent-noise VAE flavor (B2); held-
out / generalization; the capacity ladder. B1 is the cheapest PC-native sampling probe.
