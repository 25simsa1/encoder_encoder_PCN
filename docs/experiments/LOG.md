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
