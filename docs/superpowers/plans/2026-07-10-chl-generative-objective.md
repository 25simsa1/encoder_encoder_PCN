# CHL/EqProp generative objective — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the shared image decode to produce the true image from the text-driven latent when run STANDALONE, via a contrastive-Hebbian (free vs clamped) two-phase update, so text-to-image sharpens toward the true image instead of the dark conditional mean. Entirely bidirectional PC.

**Architecture:** A new opt-in `--train-mode chl` for `COCO64_GEN`. Per generative batch: text-drive the shared latents and hold them fixed; run a FREE phase (image free, standalone generation) and a CLAMPED phase (image clamped to the true image), both relaxing only the decode intermediates from the SAME fixed latent; update the decode by the contrast (learn the clamped config with `+gen_lr`, anti-learn the free config with `−gen_lr`, weight decay off). Interleaved with the recon step, warm-started from the recon model.

**Tech Stack:** TensorFlow 2.21, the existing PCN layer classes and their local `update_wts`/`update_b`, `tools/clusterrun.sh` + detached `sbatch` (L4 preferred; H200 was drained).

## Global Constraints

- Bidirectional PC ONLY. Both phases are the model's own `update_state` relaxation on the one shared-weight net; only the image clamp differs. The decode is the top-down direction of the shared weights, NOT a separate net. The true image and caption enter ONLY as clamps. The weight update is the class's own local `update_wts`/`update_b` applied at the two equilibria (CHL is local) — NO backprop, NO global loss, NO Adam/momentum, no change to per-layer math.
- The five shared latents are held FIXED (unclamped but excluded from relax/weight loops) and IDENTICAL across both phases, so the contrast varies only the image.
- NATIVE_7B untouched (a COCO64_GEN train mode; `--train-mode recon` default byte-identical). NATIVE keeps `GATE_MATCH nlayers=143`. Never relax the gate.
- Stable recipe (recon lr 1e-3, recon wd 3e-2, state_clip 400, gelu on stride-1 convs). CHL weight step uses a gentle `gen_lr` with weight decay OFF.
- If CHL destabilizes (energy climbing to the clip, `max_abs_state` toward 400, DIVERGED, or a clamp/guard RuntimeError), STOP and check in with the user before changing tack (same discipline as the earlier B-gate).
- Per the repo CLAUDE.md: after a run/work chunk, append a dated entry to `docs/experiments/LOG.md` and update `docs/STATE.md`. First-person student commits, no AI attribution. Commit locally; controller pushes at checkpoints. `clusterrun` cannot take an inline `python3 -c` with single quotes.

## File Structure

- `train_coco64.py` (Task CHL.1) — add `"chl"` to `--train-mode`, a `--gen-relax-k0` arg, a module-level `chl_step(...)`, and the interleave branch.
- (Task CHL.2) — warm-started CHL retrain + darkness diagnostic + text-to-image retest (controller-driven run, no code commit).

---

## Task CHL.1: the CHL step and train-mode wiring

**Files:**
- Modify: `train_coco64.py`
- Test: (cluster) CHL stability smoke

**Interfaces:**
- Consumes: `m._shared_latent_pairs`, `m._image_path_layers` (already on the model), `update_wts`/`update_b`.
- Produces: `--train-mode {recon,gen,chl}`, `--gen-relax-k0` (default None → `--relax`), and `chl_step(m, img_np, txt_np, mask_np, k0, k1, k2, gen_lr)`.

- [ ] **Step 1: add `"chl"` to the train-mode choices and a `--gen-relax-k0` arg.**
```python
    ap.add_argument("--train-mode", default="recon", choices=["recon", "gen", "chl"])
```
and, next to the other gen-relax args:
```python
    ap.add_argument("--gen-relax-k0", type=int, default=None)   # phase-0 (set the latent) relax steps
```

- [ ] **Step 2: add the module-level `chl_step`** (next to `generative_step`, above `main`):
```python
def chl_step(m, img_np, txt_np, mask_np, k0, k1, k2, gen_lr):
    """Contrastive-Hebbian generative step (PC-native, no backprop, no separate decoder).
    Phase 0: caption clamped, image=zeros unclamped, relax k0 so the shared latents become
    text-set; then HOLD the latents fixed (unclamped but excluded from the decode loops) for
    both phases below, so the contrast varies only the image.
    FREE phase: latents fixed, image FREE, relax the decode k1 -> the standalone generation;
    anti-learn it (decode weight step with -gen_lr, weight decay off).
    CLAMPED phase: same fixed latents, img_input clamped to the TRUE image, relax the decode
    k2 -> the target; learn it (decode weight step with +gen_lr, weight decay off).
    Net decode update: wts -= gen_lr*(g_clamped - g_free). Ends in the recon clamp config."""
    T = tf.convert_to_tensor
    img = T(img_np); txt = T(txt_np); mask = T(mask_np)
    pairs = m._shared_latent_pairs
    latent_ids = set()
    for a_, b_ in pairs:
        latent_ids.add(id(a_)); latent_ids.add(id(b_))
    decode = [L for L in m._image_path_layers
              if hasattr(L, "update_wts") and id(L) not in latent_ids and L is not m.img_input]

    def _weight_step(signed_lr):
        # local weight step on the decode with a signed rate and weight decay OFF, restored after
        orig = [(L, L.learning_rate, getattr(L, "bias_lr", None), getattr(L, "weight_decay", None)) for L in decode]
        for L in decode:
            L.learning_rate = signed_lr
            if hasattr(L, "bias_lr"):
                L.bias_lr = signed_lr
            if hasattr(L, "weight_decay"):
                L.weight_decay = 0.0
        for L in decode:
            L.update_wts(); L.update_b()
        for L, lr0, blr0, wd0 in orig:
            L.learning_rate = lr0
            if blr0 is not None:
                L.bias_lr = blr0
            if wd0 is not None:
                L.weight_decay = wd0

    # Phase 0: text-drive the shared latents (image zeros, unclamped, full relax)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(tf.zeros_like(img), txt, mask)
    m.img_input.is_clamped = False
    for _ in range(k0):
        for L in m.trainable_layers:
            L.update_state()
        m.img_input.update_state()
    # latents are now text-set; they stay unclamped but are excluded from the decode loops (fixed)

    # FREE phase: latents fixed, image FREE, relax the decode, then ANTI-LEARN
    m.img_input.is_clamped = False
    for _ in range(k1):
        for L in decode:
            L.update_state()
        m.img_input.update_state()
    _weight_step(-gen_lr)

    # CLAMPED phase: latents fixed, img_input clamped to the TRUE image, relax the decode, then LEARN
    m.img_input.set_state(img); m.img_input.is_clamped = True
    for _ in range(k2):
        for L in decode:
            L.update_state()
    _weight_step(+gen_lr)

    # restore the recon clamp config for the next recon step
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
```

- [ ] **Step 3: interleave the CHL step in the per-batch loop.** Extend the existing gen branch (after the recon step, before `step += 1`):
```python
            if a.train_mode == "gen" and step % a.gen_every == 0:
                generative_step(m, img[bi], txt[bi], mask[bi],
                                a.gen_relax_k1 or a.relax, a.gen_relax_k2 or a.relax, a.gen_lr)
            elif a.train_mode == "chl" and step % a.gen_every == 0:
                chl_step(m, img[bi], txt[bi], mask[bi],
                         a.gen_relax_k0 or a.relax, a.gen_relax_k1 or a.relax, a.gen_relax_k2 or a.relax,
                         a.gen_lr or a.lr)
```
`--train-mode recon` (default) and `gen` are unchanged; `chl` runs the new step. `gen_lr or a.lr` guarantees a non-None rate for the sign-flip.

- [ ] **Step 4: CHL stability smoke** (cluster, Bash tool timeout 600000 ms; use `--gpu L4`, node n7/n8, since H200/n15 was drained). First `python3 -c "import ast; ast.parse(open('train_coco64.py').read())"`. Then:
```
tools/clusterrun.sh --name chl_smoke --gpu L4 --mem 40G --cpus 4 --time 00:25:00 --sync "train_coco64.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py coco64_data.py infonce.py" --run "python3 train_coco64.py --config coco64_gen --train-mode chl --gen-lr 3e-4 --pairs 64 --epochs 4 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --energy-every 5 --ckpt ckpt_chl_smoke"
```
Expect: `config=coco64_gen`, energy lines finite, `max_abs_state` under 400, NO `DIVERGED`/NaN, `TRAIN_DONE`, and NO clamp-signature `RuntimeError` from `update_states_wts_b_relaxed` (the CHL step must restore the recon clamp config so the compiled recon sweep's guard holds). This smoke checks the two-phase step runs, the lr/wd/clamp hygiene restores, and stability — NOT generation quality. Leave `ckpt_chl_smoke*` unstaged.

- [ ] **Step 5: Commit**
```bash
git add train_coco64.py
git commit -m "added --train-mode chl: a contrastive-Hebbian generative step that contrasts a free standalone generation against a clamped-to-true-image pass from the same held-fixed text-set latent, updating the decode by the difference (anti-learn free, learn clamped, weight decay off). pure local PC rule; recon/gen paths unchanged"
```

---

## Task CHL.2: warm-started CHL retrain + retest (deliverable)

**Files:** none (runs CHL.1's script + `tools/darkness_diag.py` + `tools/gen_retest.py`).

- [ ] **Step 1: launch the warm-started CHL retrain** (detached `sbatch`, L4). Seed a fresh ckpt dir from the clean recon best (`cp -r ckpt_gen_best ckpt_chl`), then run `train_coco64.py --config coco64_gen --train-mode chl --gen-lr 3e-4 --gen-every 4 --pairs 2000 --epochs 30 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --energy-every 50 --ckpt ckpt_chl --resume`. Monitor energy + `max_abs_state` (the CHL contrast can be unstable). If it destabilizes, STOP and check in with the user before changing tack (do not silently retune into a different approach). If stable, let it finish; if a stable pre-drift checkpoint is the best available, keep that.

- [ ] **Step 2: darkness diagnostic + text-to-image retest** on the CHL checkpoint. Run `tools/darkness_diag.py --ckpt ckpt_chl --k 8` (does the standalone-decode brightness rise — gen/true mean ratio moving from ~0.4 toward 1.0 — and are the fine-scale latents no longer attenuated?) and `tools/gen_retest.py --ckpt ckpt_chl --k 8 --relax 150` with `--fetch`, then read a few `t2i_*` PNGs. THE KEY check: does the STANDALONE generation (and the boosted one) now show caption-specific STRUCTURE with fuller brightness/contrast, versus the prior dark banded fields, with reconstruction and image-to-caption still intact.

- [ ] **Step 3: record the outcome** — append a dated entry to `docs/experiments/LOG.md` and update `docs/STATE.md`: CHL stability, the brightness ratio and latent-attenuation change, whether standalone text-to-image sharpened toward recognizable, and recon/i2t status. No code commit; the deliverable is the checkpoint plus the verdict on whether the CHL objective produces sharper text-to-image. If it does not, that further localizes the obstacle (e.g. the LARS-scaled realization, or a genuinely harder generative-modeling gap), which is the next question.

---

## Plan exit criteria

An opt-in CHL generative mode shipped (NATIVE unchanged, `GATE_MATCH nlayers=143`; recon default byte-identical), COCO64_GEN retrained with it (stably or with a captured pre-drift checkpoint), and a clear read on whether contrastive-Hebbian training sharpens standalone text-to-image toward the true image. The raw-gradient fallback, held-out, and the capacity ladder are out of scope here.

## Self-Review

- **Spec coverage:** Component "Approach" (free/clamped contrast, latent held fixed) = CHL.1 Step 2 (`chl_step` phase 0 + free + clamped); "Implementation" (sign-flip via signed lr, wd off, decode set) = Step 2's `_weight_step` + `decode`; interleave = Step 3; validation = Step 4 (NATIVE untouched + smoke) and CHL.2 (retrain + darkness + retest). No gaps.
- **Placeholder scan:** all code and commands concrete; `gen_lr or a.lr` guarantees a rate; CHL.2 is an explicit run + the existing diagnostics, not a placeholder.
- **Type/name consistency:** `chl_step(m, img_np, txt_np, mask_np, k0, k1, k2, gen_lr)` defined and called with matching arity (k0=`--gen-relax-k0`, k1/k2 reuse `--gen-relax-k1/k2`, gen_lr=`--gen-lr or --lr`); `"chl"` added to `--train-mode` choices; `decode`/`latent_ids` built from `_image_path_layers`/`_shared_latent_pairs` as in the reviewed `generative_step`; the sign-flip reuses the negative-`learning_rate` path already validated for `--gen-lr`.
