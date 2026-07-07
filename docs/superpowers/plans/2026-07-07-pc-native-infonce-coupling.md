# PC-native InfoNCE Coupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, PC-native InfoNCE coupling that aligns the image-branch and text-branch codes at the deepest shared scale, to make text→image generation work, without any backprop through the network.

**Architecture:** Compute the InfoNCE contrastive gradient ONLY w.r.t. the two 100-dim branch codes (`inter2` image, `inter12` text), inject it as an extra error on those code states during an eager relax loop, then take the existing local LARS weight step. The compiled sweeps and NATIVE stay untouched (InfoNCE is a separate eager training path used only when `--infonce-lambda > 0`).

**Tech Stack:** TensorFlow 2.21, the Colby H200 via `tools/clusterrun.sh`, `~/tf-env`, pytest for the local gradient test. Builds on the gelu + wd=3e-2 + state_clip=400 COCO64 config.

## Global Constraints

- Learning stays predictive coding. The InfoNCE gradient is computed w.r.t. the CODES only (a `tf.GradientTape` scoped strictly to `codes -> loss`, never touching network weights) and injected as a relaxation error; WEIGHT updates remain the existing local beta-less LARS. NO backprop through the network, NO Adam/momentum.
- InfoNCE is opt-in: `--infonce-lambda 0` (default) reproduces current behavior. NATIVE_7B must still `GATE_MATCH nlayers=143`.
- The five shared-latent pairs stay aliased. Contrast the DEEPEST shared pair only (`inter2`/`inter12`).
- GPU runs via `tools/clusterrun.sh` on the H200; do NOT pass inline `python3 -c` with single quotes (use a script). Commits: first-person student, NO AI attribution / Co-Authored-By / "Generated with". Commit locally; controller pushes at checkpoints.

## File Structure

- Create `infonce.py`: the pure `infonce_grads` function (contrastive loss + code gradients + retrieval accuracy). One responsibility, no model/TF-graph coupling.
- Modify `encoder_encoder_pcn.py`: store references to the deepest-scale branch-code layers in `__init__` (`self._infonce_codes = (inter2, inter12)`). Inert (no wiring/width change).
- Modify `train_coco64.py`: `--infonce-lambda` / `--infonce-tau` args and an eager InfoNCE training path (relax + inject + LARS step) with alignment-metric logging.
- Create `tests/test_infonce.py`: local unit test for `infonce_grads` (no GPU).

---

### Task 1: `infonce_grads` pure function

**Files:**
- Create: `infonce.py`
- Test: `tests/test_infonce.py`

**Interfaces:**
- Produces: `infonce_grads(u, v, tau=0.07) -> (du, dv, acc, loss)` where `u,v` are `(B,D)` code tensors, `du,dv` are `(B,D)` gradients of the symmetric InfoNCE loss w.r.t. `u,v` (through L2-normalization), `acc` is scalar batch retrieval accuracy (fraction where the matched pair is the argmax), `loss` is scalar.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infonce.py
import numpy as np, tensorflow as tf
from infonce import infonce_grads

def test_perfect_alignment_high_acc_low_loss():
    u = tf.constant(np.eye(4, 8), dtype=tf.float32)   # 4 distinct codes
    v = tf.constant(np.eye(4, 8), dtype=tf.float32)   # identical -> matched pairs align
    du, dv, acc, loss = infonce_grads(u, v, tau=0.07)
    assert float(acc) == 1.0
    assert float(loss) < 0.1
    assert du.shape == (4, 8) and dv.shape == (4, 8)

def test_misaligned_lower_acc_grad_pulls_together():
    rng = np.random.default_rng(0)
    u = tf.constant(rng.normal(size=(6, 8)), dtype=tf.float32)
    v = tf.constant(rng.normal(size=(6, 8)), dtype=tf.float32)   # random -> unaligned
    du, dv, acc, loss = infonce_grads(u, v, tau=0.07)
    assert float(loss) > 0.5                       # random pairs are hard
    assert bool(tf.reduce_all(tf.math.is_finite(du))) and bool(tf.reduce_all(tf.math.is_finite(dv)))
    # a gradient DESCENT step on u should reduce the loss
    u2 = u - 0.5 * du
    _, _, _, loss2 = infonce_grads(u2, v, tau=0.07)
    assert float(loss2) < float(loss)
```

- [ ] **Step 2: Run, verify fail** — `cd ~/encoder_encoder_PCN && python3 -m pytest tests/test_infonce.py -q` → FAIL (no module).

- [ ] **Step 3: Write `infonce.py`**

```python
"""PC-native InfoNCE: symmetric contrastive loss over a batch of paired codes, with
gradients taken ONLY w.r.t. the codes (a GradientTape scoped to codes->loss; it never
touches network weights). Used to inject a coupling error into PC relaxation."""
import tensorflow as tf

def infonce_grads(u, v, tau=0.07):
    u = tf.convert_to_tensor(u); v = tf.convert_to_tensor(v)
    with tf.GradientTape() as t:
        t.watch([u, v])
        un = tf.math.l2_normalize(u, axis=1)
        vn = tf.math.l2_normalize(v, axis=1)
        logits = tf.matmul(un, vn, transpose_b=True) / tau      # (B,B)
        B = tf.shape(logits)[0]
        labels = tf.range(B)
        loss = 0.5 * (
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tf.transpose(logits))))
    du, dv = t.gradient(loss, [u, v])
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels), tf.float32))
    return du, dv, acc, loss
```

- [ ] **Step 4: Run, verify pass** — `python3 -m pytest tests/test_infonce.py -q` → 2 pass.

- [ ] **Step 5: Commit**

```bash
git add infonce.py tests/test_infonce.py
git commit -m "added a pure InfoNCE gradient helper: symmetric contrastive loss over paired codes, gradients taken only w.r.t. the codes (no network backprop), plus batch retrieval accuracy"
```

---

### Task 2: store deepest-scale branch-code references in the model

**Files:**
- Modify: `encoder_encoder_pcn.py` (`__init__`, after the text path is built)
- Test: (cluster) NATIVE GATE_MATCH + a ref-shape check

**Interfaces:**
- Produces: `self._infonce_codes = (inter2, inter12)` — the image-branch bottleneck feeding `dense2` and the text-branch bottleneck feeding `dense4` (both `DensePCNLayer(inter_dim)`), at the deepest shared scale.

- [ ] **Step 1: Store the references.** In `__init__`, `inter2` is the image bottleneck created just before `dense2` (`dense2 = DensePCNLayer(shared_dim, ..., inter2)`), and `inter12` is the text bottleneck created just before `dense4` (`dense4 = DensePCNLayer(shared_dim, ..., inter12, share_state_layer=dense2)`). After both exist, add:
```python
        self._infonce_codes = (inter2, inter12)   # deepest-scale image / text branch codes (for optional InfoNCE coupling)
```
This is inert: it stores references, adds no layer, changes no wiring or width.

- [ ] **Step 2: NATIVE GATE_MATCH + ref check** (H200, Bash tool timeout 600000 ms). Create `tools/_ref_check.py`:
```python
from encoder_encoder_pcn import EncoderEncoderPCN
m = EncoderEncoderPCN(1e-4)
a, b = m._infonce_codes
print("REFS", type(a).__name__, a.num_units, type(b).__name__, b.num_units)
```
Run: `tools/clusterrun.sh --name infonce_ref --gpu H200 --mem 96G --cpus 4 --time 00:30:00 --sync "encoder_encoder_pcn.py pcn_config.py tools/_ref_check.py tools/rewrite_gate.py tools/gate_compare.py" --run "python3 tools/_ref_check.py && python3 tools/rewrite_gate.py --steps 2 --save golden_infonce.npz && python3 tools/gate_compare.py golden_baseline.npz golden_infonce.npz"`
Expected: `REFS DensePCNLayer 100 DensePCNLayer 100` (both are the 100-dim `inter_dim` bottlenecks) and `GATE_MATCH nlayers=143` (storing refs is inert). Delete the throwaway after. If `num_units` is not 100, the wrong layers were captured — fix the references.

- [ ] **Step 3: Commit**

```bash
git add encoder_encoder_pcn.py
git commit -m "stored references to the deepest-scale image/text branch-code layers (inter2/inter12) for the optional InfoNCE coupling. inert: NATIVE still GATE_MATCHes"
```

---

### Task 3: eager InfoNCE training path in `train_coco64.py`

**Files:**
- Modify: `train_coco64.py`
- Uses: `infonce_grads` (Task 1), `m._infonce_codes` (Task 2).

**Interfaces:**
- Produces: `--infonce-lambda` (default 0.0) and `--infonce-tau` (default 0.07) args. When lambda>0, training uses the eager InfoNCE path and logs `infonce_acc` / `infonce_loss` alongside energy.

- [ ] **Step 1: Add the args + the eager InfoNCE batch step.** Add `from infonce import infonce_grads` and:
```python
    ap.add_argument("--infonce-lambda", type=float, default=0.0)
    ap.add_argument("--infonce-tau", type=float, default=0.07)
```
Add this function (module level, above `main`):
```python
def infonce_relax_step(m, img, txt, mask, relax, lam, tau):
    """Relax-then-step with an InfoNCE error injected at the deepest branch codes each
    relax substep, then the existing local LARS weight step. Eager (does not use the
    compiled sweep). Returns (infonce_acc, infonce_loss)."""
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    ui, vi = m._infonce_codes
    acc = tf.constant(0.0); loss = tf.constant(0.0)
    for _ in range(relax):
        for L in m.trainable_layers:
            L.update_state()
        du, dv, acc, loss = infonce_grads(ui.state, vi.state, tau)
        ui.state.assign_sub(ui.state_lr * lam * du)
        vi.state.assign_sub(vi.state_lr * lam * dv)
    for L in m.trainable_layers:
        L.update_wts(); L.update_b()
    return float(acc), float(loss)
```

- [ ] **Step 2: Branch the training loop on lambda.** In `main`, after building/config-ing the model, in the per-batch loop replace the single `update_states_wts_b_relaxed(...)` call with:
```python
            if a.infonce_lambda > 0:
                ia, il = infonce_relax_step(m, tf.convert_to_tensor(img[bi]), tf.convert_to_tensor(txt[bi]),
                                            tf.convert_to_tensor(mask[bi]), a.relax, a.infonce_lambda, a.infonce_tau)
            else:
                m.update_states_wts_b_relaxed(num_weight_steps=1, num_relax_steps=a.relax)
```
And in the energy-logging block, when `a.infonce_lambda > 0`, also print `infonce_acc={ia:.3f} infonce_loss={il:.4f}` (keep the last `ia,il`).

- [ ] **Step 3: Cluster smoke** (H200, Bash tool timeout 600000 ms): `tools/clusterrun.sh --name infonce_smoke --gpu H200 --mem 64G --cpus 4 --time 00:25:00 --sync "train_coco64.py infonce.py coco64_data.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py" --run "python3 train_coco64.py --pairs 64 --epochs 4 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --infonce-lambda 0.1 --relax 15 --batch 8 --energy-every 5 --ckpt ckpt_infonce_smoke"`
Expected: energy lines that ALSO show `infonce_acc=` and `infonce_loss=`, states finite, `TRAIN_DONE`, and `infonce_acc` trending UP (the codes aligning) / `infonce_loss` down over the few steps. No NaN/DIVERGED. (64 pairs, fast.)

- [ ] **Step 4: Commit**

```bash
git add train_coco64.py
git commit -m "added an opt-in eager InfoNCE training path: inject the contrastive code-gradient at the deepest branch codes each relax substep, then the usual local LARS weight step. logs retrieval accuracy; lambda=0 keeps the old path"
```

---

### Task 4: COCO64 InfoNCE run + text→image retest (deliverable)

**Files:** none (runs Task 3's script + a generation retest).

- [ ] **Step 1: Launch the 2k InfoNCE run** (detached sbatch on the H200; ~hours eager, so checkpoint + a modest epoch budget). Sync the files, then submit:
`sbatch -p gpu --gres=gpu:H200:1 -c 4 --mem=96G -t 08:00:00 -J p4_infonce -o infonce_%j.log --wrap "<env exports> ; cd $HOME/encoder_encoder_PCN; python3 train_coco64.py --pairs 2000 --epochs 15 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --infonce-lambda 0.1 --infonce-tau 0.07 --relax 15 --batch 8 --energy-every 50 --ckpt ckpt_infonce"` (env exports = the PATH/LD_LIBRARY_PATH/PYTHONPATH block used in the other runs). Monitor with the Monitor tool: watch `infonce_acc` (should rise) + energy (should still descend) + no divergence. If `infonce_acc` stays at chance (~1/batch) or training diverges, tune `--infonce-lambda` (try 0.03, 0.3) and/or `--infonce-tau` and re-run; record what was tried.

- [ ] **Step 2: text→image generation retest** on `ckpt_infonce_best`. Write a small script (like the prior gen probes, conv-activation gelu, 150 relax steps) that, for a few in-sample pairs, generates caption→image (save PNG) and image→caption, and prints the generated-image PNG sizes + a caption sample. Run via `clusterrun` with `--fetch`. The KEY check: do the text→image PNGs now show STRUCTURE (size well above the ~91-byte uniform-blob floor, visibly non-uniform) rather than blobs, while image→caption + reconstruction stay intact. Read a couple fetched PNGs to judge visually. Delete the throwaway script after.

- [ ] **Step 3: Record the outcome** in an SP report: final `infonce_acc`, whether text→image moved off the blob (PNG sizes + a visual read), and any lambda/tau tuning. No code commit; the deliverable is the checkpoint + the verdict on whether PC-native InfoNCE unblocks text→image. If it does not move text→image even with alignment rising, that is itself the finding (coupling-the-codes is necessary-not-sufficient), consistent with the functional-version experience.

---

## Plan exit criteria

A working, opt-in PC-native InfoNCE coupling (weights still learned only by local LARS; NATIVE GATE_MATCH intact), trained on COCO64 with the image/text code alignment rising (`infonce_acc` up), and a clear read on whether text→image generation moves off the uniform blob. Multi-scale InfoNCE, warm-up scheduling, and held-out are follow-ups.

## Self-Review

- Spec coverage: Component 1 (codes to contrast, `inter2`/`inter12`, deepest scale) = Task 2 refs + Task 3 use; Component 2 (closed-form/code-scoped InfoNCE gradient, no net backprop) = Task 1 `infonce_grads` (tape scoped to `codes->loss`); Component 3 (inject into relaxation + local LARS learns, joint, opt-in) = Task 3 eager path; Component 4 (NATIVE GATE_MATCH + text→image retest + alignment metric) = Task 2 gate, Task 3 acc logging, Task 4 run + retest. No gaps.
- Placeholder scan: all code is complete; the lambda/tau tuning in Task 4 is an explicit sweep with named values (0.03/0.1/0.3), not a placeholder; throwaway probes are described and deleted, not committed.
- Type consistency: `infonce_grads(u, v, tau) -> (du, dv, acc, loss)` is defined in Task 1 and consumed with that exact signature in Task 3; `self._infonce_codes = (inter2, inter12)` defined in Task 2 and read as `ui, vi = m._infonce_codes` in Task 3; `--infonce-lambda`/`--infonce-tau` consistent across Tasks 3-4.
