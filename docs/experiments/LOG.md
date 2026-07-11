# Experiment log

Newest on top. One entry per run or outcome. Never edit past entries.

## Entry format
```
### YYYY-MM-DD short title
- config or command, <what was run>
- result, <the number or observation>
- takeaway, <one line>
```

---

### 2026-07-11 weight-norm stabilizer, layers + COCO64 inertness gate (byte-identical when off)
- config or command, implemented the PC-native weight-normalization reparameterization on the conv (c0a4d42) and dense (e0ba22e) layers (w = g_mag * v/||v|| per output unit, weight() accessor both directions, update_wts split into a radial magnitude step + a tangential direction step, LARS dropped for these layers, opt-in --weight-norm); 7 unit tests pass. Gate on L4, tools/rewrite_gate.py --config coco64 --relaxed vs the pre-change banked ref docs/superpowers/gate_ref_coco64.npz
- result, byte-identical when off CONFIRMED: run-to-run GATE_MATCH (current code deterministic on one node) AND ref-vs-cur GATE_MATCH nlayers=88 tol=1e-4. First gate attempt on node n7 showed a spurious GATE_MISMATCH of 1.27-1.32e-4 on 3 of 88 layers = cross-node L4 fp variation (cuDNN), NOT the code; re-running on a node consistent with the ref gave the exact match. Operational lesson, a single-run L4 GATE_MISMATCH at ~1e-4 needs a same-node run-to-run re-check before it counts as a regression (or pin the node / use tol ~2e-4 cross-node). NATIVE-143 gate deferred to a big-GPU window (H200 drained); interim proof = this COCO64 gate + provable inertness (weight() returns self.wts, else-LARS unchanged) + the unit tests
- takeaway, the weight-norm change is inert when off (NATIVE-safe), so the structural fix is in place; next is wiring --weight-norm into train_coco64 (with g_mag persistence) and the make-or-break CHL retrain that must hold training past the ep13 norm-inflation wall

---

### 2026-07-11 CHL destabilizes ~ep13 like every approach: NORM INFLATION is the common blocker
- config or command, CHL retrain (job 8830, warm-start recon, --train-mode chl --gen-lr 3e-4 --gen-every 4, 30 ep, L4); darkness_diag + gen_retest on ckpt_chl (ep29, corrupted) and ckpt_chl_best (stable, early)
- result, CHL held to ~ep10 (energy 0.014, state ~110) then norm-inflated: state hit the 400 clip by ep16-17, energy exploded to 7-10 by ep19-20. Stable best ckpt: standard test_step still a blob; boosted generation has the best CONTRAST yet (std ratio 0.40 vs prior 0.08-0.30) but the same dark brightness (mean ratio 0.38), not recognizable. Monitoring gap: the B-gate grep watched only DIVERGED/RuntimeError, not state-pinned-at-clip, so the Monitor reported a clean TRAIN_DONE on a blown-up run
- takeaway, EVERY generative-training approach (InfoNCE, one-sided bridge, gentle-lr, CHL) destabilizes via NORM INFLATION at ~ep13, capping the achievable generation. The objective is not the blocker; the underlying norm-inflation instability is (the beta-less-LARS / effective-LR-vs-scale thread). The real next lever is the deferred structural stabilizer (muP effective-LR scaling, weight normalization, or a norm penalty in the PC energy) so any generative training can run long enough to sharpen

---

### 2026-07-10 objective diagnostic: darkness is the top-down conditional-mean, not weight decay
- config or command, tools/objective_diag.py A/B on ckpt_gen_best (clean recon), 30 generative steps with weight decay ON (3e-2) vs OFF (0) on the decode layers, measuring decode weight-norm + boosted-generation brightness
- result, decode weight-norm unchanged in both arms (424.5 -> 424.3, ratio 1.000, so weight decay is NOT shrinking the weights); weight-decay-OFF ended darker (gen mean 0.033) than ON (0.087), refuting the shrinkage hypothesis. Even the clean recon decode already generates dark top-down (mean 0.155, 0.40x true) with ZERO generative steps, and generative steps rearrange the weights toward an even darker output
- takeaway, the darkness is the top-down PC decode producing the low-amplitude CONDITIONAL MEAN (the blurry average image consistent with the latent); the current bridge objective (true image clamped at the bottom during the weight step) does not teach a strong latent->image map usable at generation. The lever is a redesign of the generative objective for sharp top-down generation, not tuning / longer training

---

### 2026-07-10 more generative training (gen-every 2) made amplitude WORSE, points at the objective
- config or command, --gen-lr 3e-4 --gen-every 2 warm-started from clean recon (job 8826); drifted (energy peaks 0.010->0.021 ep7-10) and cancelled; darkness retest on the ep8 ckpt (~1000 gen steps, more than gentler-A's 750)
- result, generated brightness 0.186x true (mean 0.071), WORSE than gentler-A's 0.386x; fine-scale text-set latents attenuated (T/I ratios 0.41, 0.57 at scales 4-5 vs ~1.0 in gentler-A). gen-every 2 was a hyperparameter mistake (doubled generative frequency -> drove the drift), so not a fair test of gentle-lr
- takeaway, more generative training via this config shrank the signal rather than sharpening it; the fine-scale attenuation suggests the phase-2 objective may be SHRINKING the decode (hypothesis, weight decay dominating the small-error weight step), a fixable objective flaw worth diagnosing before any longer run

---

### 2026-07-10 darkness diagnostic: the gap is decode-amplitude, not the latent or text path
- config or command, tools/darkness_diag.py on ckpt_gen_warm (gen-trained ep12), compare text-set vs image-set shared-latent scale + generated-image brightness vs true
- result, text-set latents match image-set in scale (norm ratio T/I = 0.89-1.12 across all 5 scales, so the text path sets the latent at the right scale). But the top-down-decoded image is mean 0.148 / std 0.080 vs the true image mean 0.383 / std 0.263 (0.39x brightness, 0.30x contrast, max only 0.357 vs 1.0). The same decode weights produce a bright image in reconstruction (image clamped), so the decode is capable but has not learned to produce full-scale output from the latent top-down
- takeaway, the darkness is a DECODE-TRAINING gap, not a latent-capacity ceiling or text-path failure; the lever is a better/longer/balanced generative-training run (keep recon at the floor, train the decode much longer) to push the output to full brightness/contrast/detail

---

### 2026-07-10 gentler-A generative training: stable, modestly shifts boosted generation, not recognizable
- config or command, gentler A (warm-start from the recon-trained COCO64_GEN best, then --train-mode gen --gen-every 4), job 8820 on L4, then text->image retests on the ep12 checkpoint
- result, trains STABLY for 12 epochs (energy near floor, state bounded ~120-160) then a slow drift crosses the thresholds at ep13 (stopped, kept the ep12 ckpt = ~750 stable generative steps). Standard test_step t2i still blobs (it uses the symmetric relaxation, not the top-down training regime). recon still works (PR 3.0) but degraded (energy 0.023 vs 0.006 floor); image->caption degraded to empty. Top-down-BOOSTED generation on the gen-trained ckpt: PR 6.38 at gamma 1.0 with ~3.4KB content, vs the recon-only model's high PR only as tiny speckle (730B) — so more caption-distinct and content-richer, but visually dark faintly-banded fields, NOT recognizable scenes (the per-caption variation is real but low-amplitude)
- takeaway, the generative-training objective is stable at a gentle schedule and modestly improves the top-down-boosted output, but does not crack recognizable text->image and costs recon+i2t; the remaining obstacle is likely the shared 100-dim latent capacity / the fundamental difficulty of from-scratch PC 64px text->image

---

### 2026-07-10 approach-A generative training destabilizes (recon-vs-generation tension)
- config or command, COCO64_GEN --train-mode gen at gen-every 1, 2k pairs, stable recipe (lr 1e-3 wd 3e-2 clip 400 gelu), job 8819 on the L4
- result, the two-phase generative step interleaves correctly (traces once, clamp hygiene holds) and is stable to ep2 (energy ~0.008, max|state| ~44), then energy climbs 0.008->0.037 and max|state| jumps 44->126 by ep3, accelerating on both; cancelled at step 900
- takeaway, the mechanism is sound (GT.1/GT.2 reviewed clean) but at gen-every 1 the recon-vs-generation pressure destabilizes the shared weights; next lever is a gentler A schedule (recon warm-up then infrequent generative steps) or the soft-nudge Approach B, a gated user decision

---

### 2026-07-09 top-down-boosted generation confirms invertible downsampling reconnected the pathway
- config or command, top-down-boosted test_step generation on COCO64_GEN/ckpt_gen_best, gamma sweep 0/0.5/1/2 (the boost now flows THROUGH the invertible strided-conv downsamplers, unlike the maxpool-blocked earlier attempt), job 8808 on the L4
- result, image PR rose 0.00 -> 1.34 -> 2.32 -> 6.93 (max 8) with gamma; the caption now drives the image per-pixel and distinctly per caption (vs the maxpool model's uniform green tint), but the outputs are transpose-conv checkerboard fields (low gamma) / caption-varying colorful speckle (high gamma), not recognizable scenes
- takeaway, invertible downsampling + a top-down generation schedule solve the structural block AND the drive-balance at inference (caption controls the image, no retraining) -- recognizable CONTENT is the one remaining lever and needs the generative-training objective, since the decode was trained both-clamped and was never top-down-self-sufficient

---

### 2026-07-08 strided-conv downsampling ships but text to image still blobs
- config or command, COCO64_GEN with strided-conv downsampling
- result, downsampling trains stably and ships clean, text to image still blobs under the standard relaxation
- takeaway, invertible downsampling is necessary but not sufficient, drive-balance and non-generative training remain the obstacles

---

### 2026-07-07 PC-native InfoNCE coupling fails to unblock text-to-image
- config or command, PC-native InfoNCE coupling at lambda 0.1, 0.3, and 1.0, validated against the no-InfoNCE baseline
- result, code alignment stayed at chance and text-to-image generation stayed a blob at every lambda
- takeaway, contrasting the paired codes alone does not fix the image-dominated shared latent, the mechanism and follow-ups are recorded for the next attempt

---

### 2026-07-07 fixed the attention pad-mask no-op SP1 review caught
- config or command, moved the mask broadcast in AttentionPCNLayer from the query axis to the key axis
- result, padded caption positions now get excluded for every query, re-gated GATE_MATCH nlayers=143 for NATIVE_7B (still inert there since its mask is all-zeros)
- takeaway, the old broadcast shape let softmax's shift-invariance silently no-op the mask, code review caught what the running tests missed

---

### 2026-07-07 best-checkpoint save reveals coco64 does overfit then destabilize
- config or command, state_clip=150 training run on coco64, checkpointed on every save
- result, energy dropped from 2.17 to 0.13 by epoch 7, then the run destabilized after about epoch 8, and max_to_keep=1 had been overwriting the good weights with the blown-up ones
- takeaway, save the best (lowest-energy) checkpoint separately so the learned model survives a later divergence

---

### 2026-07-07 state-clip added after a state-norm divergence
- config or command, --state-clip arg capping max |state| after relaxation on the dense and conv layers carrying the shared latents
- result, the lower-learning-rate run still diverged by state-norm inflation, the same failure mode seen at 7.7B, the lower rate only delayed it
- takeaway, cap the runaway quantity itself (state norm) rather than trying to out-tune the learning rate around it

---

### 2026-07-07 COCO64_156M validated on the H200
- config or command, instantiate, batch, and generate checks for COCO64_156M on H200 hardware
- result, batches produce finite states, all 5 shared-latent aliases hold, and generation works in both directions
- takeaway, the bidirectional class now has a working 64px config, clearing the way for the coco64 training work

---

### 2026-07-07 COCO64_156M tuned to land near its 156M target
- config or command, added a param counter and a --config flag to the gate tool, then tuned COCO64_156M's widths
- result, the config lands near 156M parameters
- takeaway, the gate tool now reports param counts per config, making future capacity targets a tuning exercise instead of a guess
