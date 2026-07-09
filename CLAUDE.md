# CLAUDE.md, encoder_encoder_PCN

Research codebase for a predictive coding network (PCN) studying whether a cross-modal coupling failure persists as the model scales from 156M to 7.7B parameters. Read this fully before acting.

## Session protocol (every session)
1. At session start, read `docs/STATE.md` and the top three entries of `docs/experiments/LOG.md` before touching code.
2. After any run or work chunk, append a dated entry to the top of `docs/experiments/LOG.md` and update `docs/STATE.md`. Never edit past LOG entries.
3. Keep the invariants below intact. If a request conflicts with one, stop and say so.

## Hard invariants (do not violate)
1. The NATIVE config stays byte-identical. Its downsampling path is maxpool. Any change that alters NATIVE output is a regression, so guard it, do not touch it.
2. OOM is expected. The model is intentionally about 28.7 GiB and needs a GPU of at least 40GB. Never resolve an OOM by shrinking the model, cutting capacity, or dropping the documented batch. Fix the real bug instead.
3. The decisive experimental bar does not move. Coupling is judged at the 8k-pair scale with latent retrieval primary against the banked bar of more than 3 in 2000. Do not soften or redefine it.
4. The golden numerical gate for NATIVE is GATE_MATCH at nlayers=143. It is the proof that a refactor changed nothing. A failing gate is a bug to fix, not a number to update.

## Configs
- NATIVE, the reference path, maxpool downsampling, must stay byte-identical.
- COCO64_GEN, strided-conv bidirectional downsampling so top-down generative drive flows.
- COCO64_156M, the 156M banked config.
Configs are defined in `pcn_config.py` and consumed by `train_coco64.py`.

## Orientation
Architecture and the file map live in `docs/ARCHITECTURE.md`. Current status lives in `docs/STATE.md`.
