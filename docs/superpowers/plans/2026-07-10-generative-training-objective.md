# Generative-training objective (text-to-image) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the shared image-decode weights to produce the true image from a text-driven shared latent, so text-to-image generation yields recognizable caption-varying scenes instead of artifacts, entirely within bidirectional predictive coding.

**Architecture:** A new opt-in training mode for `COCO64_GEN` interleaves the existing both-clamped recon step with a new two-phase generative step. Phase 1 clamps the caption and unclamps a zero image and relaxes so the five shared latents become text-driven. Phase 2 clamps the true image, leaves the latents unclamped but does not relax them (fixed top-down sources), relaxes the image decode intermediates to bridge, then takes the existing local weight step on those intermediates. Only clamp plus the model's own relaxation plus the existing local `update_wts`/`update_b`. No backprop, no separate decoder.

**Tech Stack:** TensorFlow 2.21 (Python 3.13), the existing PCN layer classes, `tools/clusterrun.sh` and detached `sbatch` (cluster), pytest where local.

## Global Constraints

- Bidirectional PC ONLY. One shared-weight net used both directions (same `predict_next` up, `predict_prev` down). The decode is the top-down direction of the same weights, NOT a second network. Learning stays the existing local beta-less LARS (`update_wts`/`update_b`); NO backprop through the net, NO autograd loss, NO Adam/SGD, NO change to per-layer update math.
- The true image and caption enter ONLY as clamps (boundary conditions), never as a differentiated loss.
- The five shared-latent aliases stay intact. In phase 2 the latents are held fixed by EXCLUDING them from the relax and weight loops, and are left UNCLAMPED (clamping a next-layer makes `update_state` skip it in the top-down block, which would remove the latent's downward drive — do not clamp the latents).
- NATIVE_7B is untouched (this is a COCO64_GEN training mode plus an inert constructor addition). NATIVE keeps `GATE_MATCH nlayers=143`. NEVER relax the gate.
- Stable recipe: lr 1e-3, weight_decay 3e-2, state_clip 400, gelu on stride-1 convs, downsamplers linear.
- Approach B (soft-nudge) is a GATED fallback. If A destabilizes, STOP and check in with the user before implementing or running B. Do not run B without explicit approval.
- Per the repo CLAUDE.md: after a run or work chunk, append a dated entry to `docs/experiments/LOG.md` and update `docs/STATE.md`.
- First-person student commits, NO AI attribution / Co-Authored-By / "Generated with". Commit locally; controller pushes at checkpoints. `clusterrun` cannot take an inline `python3 -c` with single quotes.

---

## File Structure

- `encoder_encoder_pcn.py` (Task GT.1) — expose `self._image_path_layers` and `self._shared_latent_pairs`. Inert (records references), NATIVE GATE_MATCH preserved.
- `train_coco64.py` (Task GT.2) — `--train-mode {recon,gen}`, `--gen-every`, `--gen-relax-k1/k2`, a module-level `generative_step(...)`, and the interleave branch in the loop.
- (Task GT.3) — retrain COCO64_GEN in gen mode plus the text-to-image retest (controller-driven run, no code commit).

---

## Task GT.1: expose the generative-step layer sets

**Files:**
- Modify: `encoder_encoder_pcn.py` (`__init__`)
- Test: (cluster) NATIVE GATE_MATCH + a reference check

**Interfaces:**
- Produces: `self._image_path_layers` (list of image-side layers) and `self._shared_latent_pairs` (list of 5 `(image_dense, text_dense)` aliased pairs).

- [ ] **Step 1: snapshot the image-path layers.** In `__init__`, immediately BEFORE the line `self.txt_input = InputPCNLayer(learning_rate)` (all image-side layers, including the five image shared latents, are appended before this), insert:
```python
        self._image_path_layers = list(self.trainable_layers)   # image side, snapshot before the text path is built
```

- [ ] **Step 2: derive the shared-latent pairs at the end of __init__.** After the entire model is constructed (after the last `share_state_layer=` alias, i.e. near the end of `__init__`), insert:
```python
        # (image_dense, text_dense) aliased shared-latent pairs, for the generative training mode
        self._shared_latent_pairs = [(L.share_state_layer, L) for L in self.trainable_layers
                                     if getattr(L, "share_state_layer", None) is not None]
```
This is inert: it records references only; it adds no layer and changes no wiring, width, or math.

- [ ] **Step 3: NATIVE gate + reference check** (cluster, H200 or any available GPU, Bash tool timeout 600000 ms). Create throwaway `tools/_gt_check.py`:
```python
import tensorflow as tf
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import COCO64_GEN as C
m = EncoderEncoderPCN(1e-4, config=C)
B = 2
img = tf.random.normal((B, C.img_resolution, C.img_resolution, 3), seed=0)
txt = tf.random.normal((B, C.txt_seq_len, C.txt_embed_dim), seed=0)
mask = tf.zeros((B, C.txt_seq_len))
m.img_input.is_clamped = True; m.txt_input.is_clamped = True
m.pass_through(img, txt, mask)
pairs = m._shared_latent_pairs
ipl_ids = set(map(id, m._image_path_layers))
print("PAIRS", len(pairs), "(expect 5)", flush=True)
print("PAIR_ALIASED", all(id(a.state) == id(b.state) for a, b in pairs), "(expect True)", flush=True)
print("IMG_LATENTS_IN_PATH", all(id(a) in ipl_ids for a, _ in pairs), "(expect True)", flush=True)
print("TXT_INPUT_EXCLUDED", id(m.txt_input) not in ipl_ids, "(expect True)", flush=True)
print("GT_CHECK_DONE", flush=True)
```
Run:
```
tools/clusterrun.sh --name gt_check --gpu H200 --mem 96G --cpus 4 --time 00:30:00 --sync "encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py coco64_data.py tools/_gt_check.py tools/rewrite_gate.py tools/gate_compare.py" --run "python3 tools/rewrite_gate.py --steps 2 --save golden_gt.npz && python3 tools/gate_compare.py golden_baseline.npz golden_gt.npz && python3 tools/_gt_check.py"
```
Expect `GATE_MATCH nlayers=143`, `PAIRS 5`, `PAIR_ALIASED True`, `IMG_LATENTS_IN_PATH True`, `TXT_INPUT_EXCLUDED True`. If the H200 (n15) is drained, use `--gpu L4` (or another available type); the check is tiny. Delete `tools/_gt_check.py` after (do not commit). If the gate mismatches, the snapshot line perturbed something; it must not.

- [ ] **Step 4: Commit**
```bash
git add encoder_encoder_pcn.py
git commit -m "exposed the image-path layer list and the 5 shared-latent aliased pairs for the generative training mode; inert reference-recording, NATIVE still GATE_MATCHes"
```

---

## Task GT.2: the generative step and the train-mode interleave

**Files:**
- Modify: `train_coco64.py`
- Test: (cluster) generative-mode stability smoke

**Interfaces:**
- Consumes: `m._shared_latent_pairs`, `m._image_path_layers` (GT.1).
- Produces: `--train-mode {recon,gen}` (default `recon`), `--gen-every` (default 1), `--gen-relax-k1/--gen-relax-k2` (default to `--relax`), and `generative_step(m, img_np, txt_np, mask_np, k1, k2)`.

- [ ] **Step 1: add the args.** In `main`'s argparse block add:
```python
    ap.add_argument("--train-mode", default="recon", choices=["recon", "gen"])
    ap.add_argument("--gen-every", type=int, default=1)
    ap.add_argument("--gen-relax-k1", type=int, default=None)
    ap.add_argument("--gen-relax-k2", type=int, default=None)
```

- [ ] **Step 2: add the module-level `generative_step`** (next to `infonce_relax_step`, above `main`):
```python
def generative_step(m, img_np, txt_np, mask_np, k1, k2):
    """PC-native generative step. Phase 1: caption clamped, image = zeros unclamped, relax k1
    so the shared latents become text-driven. Phase 2: clamp the TRUE image; leave the latents
    UNCLAMPED but do NOT relax them (fixed text-driven top-down sources), relax the image decode
    intermediates k2 to bridge, then the existing local weight step. Only clamp + update_state +
    update_wts; no backprop, no separate decoder. Ends in the recon clamp configuration."""
    T = tf.convert_to_tensor
    img = T(img_np); txt = T(txt_np); mask = T(mask_np)
    pairs = m._shared_latent_pairs
    latent_ids = set()
    for a_, b_ in pairs:
        latent_ids.add(id(a_)); latent_ids.add(id(b_))
    decode = [L for L in m._image_path_layers
              if hasattr(L, "update_wts") and id(L) not in latent_ids and L is not m.img_input]

    # Phase 1: text-drive the shared latents (image = zeros, unclamped)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(tf.zeros_like(img), txt, mask)
    m.img_input.is_clamped = False
    for _ in range(k1):
        for L in m.trainable_layers:
            L.update_state()
        m.img_input.update_state()

    # Phase 2: clamp the true image; latents stay unclamped but fixed (not relaxed);
    # relax the decode intermediates, then the local weight step
    m.img_input.set_state(img); m.img_input.is_clamped = True
    for _ in range(k2):
        for L in decode:
            L.update_state()
    for L in decode:
        L.update_wts(); L.update_b()

    # restore the recon clamp config (image+text clamped, latents unclamped) for the next recon step
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
```

- [ ] **Step 3: interleave the generative step in the per-batch loop.** Immediately AFTER the existing recon step (the `if a.infonce_lambda > 0: ... else: m.update_states_wts_b_relaxed(...)` block) and BEFORE `step += 1`, insert:
```python
            if a.train_mode == "gen" and step % a.gen_every == 0:
                generative_step(m, img[bi], txt[bi], mask[bi],
                                a.gen_relax_k1 or a.relax, a.gen_relax_k2 or a.relax)
```
The recon step keeps the encoder, latents, text path, and image-to-caption healthy; the generative step trains the decode. `--train-mode recon` (default) skips the branch entirely, so existing runs are byte-identical.

- [ ] **Step 4: generative-mode stability smoke** (cluster, Bash tool timeout 600000 ms). First `python3 -c "import ast; ast.parse(open('train_coco64.py').read())"`. Then:
```
tools/clusterrun.sh --name gen_train_smoke --gpu H200 --mem 96G --cpus 4 --time 00:25:00 --sync "train_coco64.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py coco64_data.py infonce.py" --run "python3 train_coco64.py --config coco64_gen --train-mode gen --pairs 64 --epochs 4 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --energy-every 5 --ckpt ckpt_gen_train_smoke"
```
(If n15/H200 is drained, use `--gpu L4`.) Expect: `config=coco64_gen`, energy lines finite, `max_abs_state` under 400, NO `DIVERGED`/NaN, and `TRAIN_DONE`. Critically, the run must NOT raise the compiled-sweep clamp-signature guard error, which would mean the clamp hygiene failed (the generative step left clamps in a non-recon configuration). This smoke checks the interleave, clamp hygiene, and stability, NOT generation quality. Leave `ckpt_gen_train_smoke*` unstaged.

- [ ] **Step 5: Commit**
```bash
git add train_coco64.py
git commit -m "added an opt-in --train-mode gen that interleaves a PC-native generative step with recon: text-drive the shared latents, then clamp the true image and relax the decode intermediates with the latents held fixed, then the local weight step. recon default unchanged"
```

---

## Task GT.3: retrain COCO64_GEN in gen mode + text-to-image retest (deliverable)

**Files:** none (runs GT.2's script + the existing `tools/gen_retest.py`).

- [ ] **Step 1: launch the gen-mode retrain** (detached `sbatch`; the stable recipe). Adapt `tools/run_gen.sh` (or a copy) to add `--train-mode gen` and a distinct `--ckpt ckpt_gen_train`, sync, submit. Monitor energy and `max_abs_state` with the Monitor tool. Watch specifically for the A-destabilization signals: energy climbing rather than descending, `max_abs_state` pinned near 400, a `DIVERGED` line, or (if a mid-run recon check is added) reconstruction degrading. IF ANY of these appear, STOP, report to the user, and get explicit approval before touching Approach B. Do not tune into B silently.

- [ ] **Step 2: text-to-image retest** on `ckpt_gen_train_best`. Run `tools/gen_retest.py --ckpt ckpt_gen_train_best --k 8 --relax 150` via `clusterrun --fetch gen_retest_out` (it already builds COCO64_GEN, sets gelu on stride-1 convs, restores, and generates both directions with the model's own `test_step`, no manual boost). Read a few fetched `t2i_*` PNGs. THE KEY check: does text-to-image now show caption-VARYING recognizable STRUCTURE (image participation ratio well above 1, PNGs that differ by caption and show scene content, not speckle), with reconstruction and image-to-caption still intact.

- [ ] **Step 3: record the outcome.** Append a dated entry to `docs/experiments/LOG.md` and update `docs/STATE.md` (repo CLAUDE.md protocol): final energy/stability, whether text-to-image produced recognizable caption-varying scenes (PR + a visual read), reconstruction and image-to-caption status. No code commit; the deliverable is the checkpoint plus the verdict on whether the generative-training objective unblocks recognizable text-to-image. If it did not (still coarse/speckle even with generation-trained weights), that localizes the remaining obstacle to the shared-latent capacity, which is the next question.

---

## Plan exit criteria

A PC-native generative-training mode shipped opt-in (NATIVE unchanged, `GATE_MATCH nlayers=143`; recon default byte-identical), COCO64_GEN retrained with it stably, and a clear read on whether text-to-image now produces recognizable caption-varying scenes. Approach B, held-out, and the capacity ladder are out of scope here.

## Self-Review

- **Spec coverage:** Component 1 (training structure, recon + generative interleave) = GT.2 Steps 1/3; Component 2 (the two-phase generative step, latents fixed-not-clamped) = GT.2 Step 2; Component 3 (layer sets) = GT.1; Component 4 (eager/compiled interleave, clamp hygiene) = GT.2 Steps 2/4; Component 5 (B gated) = Global Constraints + GT.3 Step 1; Component 6 (validation) = GT.1 Step 3 + GT.2 Step 4 + GT.3 Step 2. No gaps.
- **Placeholder scan:** all code and commands are concrete; `--gen-relax-k1/k2` default to `--relax` via `a.gen_relax_k1 or a.relax`; GT.3 is an explicit run plus the existing retest, not a placeholder.
- **Type/name consistency:** `self._image_path_layers` and `self._shared_latent_pairs` defined in GT.1 and consumed by name in GT.2's `generative_step`; `--train-mode`/`--gen-every`/`--gen-relax-k1`/`--gen-relax-k2` defined in GT.2 Step 1 and used in Steps 2/3; `generative_step(m, img_np, txt_np, mask_np, k1, k2)` defined and called with matching arity. The latents are left unclamped and merely excluded from the loops (matches the corrected spec), so no clamp-skip regression.
