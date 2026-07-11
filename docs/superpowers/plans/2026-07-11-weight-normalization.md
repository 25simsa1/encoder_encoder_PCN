# Weight-normalization norm-inflation stabilizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reparameterize each opt-in conv/dense layer's weight as a per-output-unit magnitude times a unit direction so the weight norm cannot run away, killing the norm-inflation instability that caps every generative objective at ~ep13, then re-run the CHL objective long enough to sharpen text-to-image.

**Architecture:** Each layer gains `w = g_mag * v/||v||` (v is the current weight tensor `wts`, g_mag a per-output-unit magnitude). A `weight()` accessor returns `w` when weight-norm is on and `self.wts` when off. The SAME `w` feeds `predict_next` (encode) and `predict_prev` (decode), so it stays one bidirectional net. `update_wts` splits the class's OWN local weight gradient into a radial magnitude update and a tangential direction update (no backprop). Opt-in via a `--weight-norm` training flag; off is byte-identical to today.

**Tech Stack:** Python, TensorFlow 2.21 (eager + the model's relaxation methods), pytest, the Colby HPC cluster (L4 nodes n7/n8) via `tools/clusterrun.sh` and detached `sbatch`.

## Global Constraints

- **Bidirectional PC only.** One shared-weight net used both directions, a local update rule, NO backprop, NO separate decoder, NO off-the-shelf optimizer. `weight()` feeds `predict_next` AND `predict_prev` identically. The update is a change of variables on the class's existing local gradient. Never drop PC to get it working.
- **NATIVE stays byte-identical.** `weight_norm` defaults False; `weight()` returns `self.wts` and `update_wts` takes the unchanged LARS branch when off. The golden gate `GATE_MATCH nlayers=143` for NATIVE_7B must still pass. `weight_norm` is a plain Python bool (so `tf.function` branches resolve at trace time, no graph change when off).
- **OOM is expected; do not shrink the model.** NATIVE needs a GPU of at least 40GB. Use L4 (n7/n8, 24GB) for COCO64_GEN and CHL. H200/n15 is DRAINED. n10 A100 is another project's run, leave it alone. Fix real bugs, never resolve OOM by cutting capacity or batch.
- **The decisive bar does not move** (8k pairs, latent retrieval > 3 in 2000). This plan does not touch it.
- **Commit style:** first-person student voice, no AI attribution, no `Co-Authored-By`, no "Generated with". Identity `Simon Sang <simonlapsang@gmail.com>`. Commit locally, push at checkpoints.
- **Writing style:** no em dashes, no colons, in any prose or docs authored here.
- **Cluster caveat:** `tools/clusterrun.sh` cannot take an inline `python3 -c` with single quotes. Put throwaway python in a file and sync it, or use double quotes carefully.
- **Docs protocol:** after any run or work chunk, append a dated entry to the TOP of `docs/experiments/LOG.md` (never edit past entries) and update `docs/STATE.md`.

---

### Task 1: Bank the pre-change gate reference

**MUST run before any layer file is edited** (Tasks 2 and 3 edit `conv_pcn_layer.py` and `dense_pcn_layer.py`; this captures the reference they will be compared against). The canonical NATIVE-143 gate needs a 40GB+ GPU (H200 drained), so this banks a COCO64_156M reference that runs on the L4 now as the runnable inertness proof. The NATIVE-143 gate is deferred to a big-GPU window (Task 4).

**Files:**
- Create (cluster-side scratch): `/tmp/gate_ref_coco64.npz` fetched to `docs/superpowers/gate_ref_coco64.npz`
- Uses: `tools/rewrite_gate.py`, `tools/gate_compare.py`, `tools/clusterrun.sh`

**Interfaces:**
- Consumes: nothing (pre-flight).
- Produces: a banked reference signature file `docs/superpowers/gate_ref_coco64.npz` that Task 4 compares against.

- [ ] **Step 1: Confirm the working tree is pre-change.** Run `git status --short conv_pcn_layer.py dense_pcn_layer.py` and confirm NEITHER layer file is modified. If either is modified, STOP and escalate (the reference would be contaminated).

- [ ] **Step 2: Generate the COCO64 reference signature on the L4.**
```
tools/clusterrun.sh --name gateref --gpu L4 --mem 40G --cpus 4 --time 00:20:00 \
  --sync "rewrite_gate.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py" \
  --run "python3 tools/rewrite_gate.py --config coco64 --relaxed --relax-steps 5 --weight-steps 2 --save /tmp/gate_ref_coco64.npz && python3 -c \"import numpy as np; d=np.load('/tmp/gate_ref_coco64.npz'); print('REF nlayers=', len(d.files))\""
```
Expect a `GOLDEN ... nlayers=<N>` line and `REF nlayers=<N>` with N > 0. Note N (the COCO64 layer count) for Task 4.

- [ ] **Step 3: Fetch the reference into the repo.** `tools/clusterrun.sh` fetches run outputs; copy the fetched `gate_ref_coco64.npz` to `docs/superpowers/gate_ref_coco64.npz`. If clusterrun did not fetch it, re-run with an explicit fetch of `/tmp/gate_ref_coco64.npz`, or `scp slsang29@hpc.colby.edu:encoder_encoder_PCN/gate_ref_coco64.npz docs/superpowers/`. Confirm the file exists locally and is non-empty.

- [ ] **Step 4: Commit the reference.**
```bash
git add docs/superpowers/gate_ref_coco64.npz
git commit -m "banked the pre-change COCO64 gate signature so I can prove the weight-norm reparameterization is byte-identical when the flag is off"
```

---

### Task 2: Conv2DPCNLayer weight normalization

**Files:**
- Modify: `conv_pcn_layer.py` (class `Conv2DPCNLayer`)
- Test: `tests/test_weight_norm.py` (conv cases)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `Conv2DPCNLayer.weight() -> tf.Tensor` (returns `self.wts` when `self.weight_norm` is False, else `g_mag * wts / ||wts||` normalized over axes [0,1,2]).
  - `Conv2DPCNLayer.enable_weight_norm() -> None` (creates `self.g_mag` = per-output-filter `||wts||`, sets `self.weight_norm = True`; raises if `wts` is None).
  - `self.weight_norm: bool` (default False), `self.g_mag: tf.Variable | None` (default None).

- [ ] **Step 1: Write the failing test.** Create `tests/test_weight_norm.py`:
```python
import numpy as np
import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer
from dense_pcn_layer import DensePCNLayer


def _realize_conv():
    L = Conv2DPCNLayer(5, (3, 3), 1e-2, "linear", padding="SAME")
    L(tf.random.normal((2, 8, 8, 3)), set_state=True)   # realizes wts
    return L


def test_conv_weight_off_is_wts_identity():
    L = _realize_conv()
    assert L.weight_norm is False
    assert L.weight() is L.wts            # off = passthrough, byte-identical


def test_conv_enable_is_seamless():
    L = _realize_conv()
    w_before = L.weight().numpy().copy()  # == wts (off)
    L.enable_weight_norm()
    assert L.weight_norm is True
    np.testing.assert_allclose(L.weight().numpy(), w_before, atol=1e-5)   # w == wts at enable
    per_unit = tf.sqrt(tf.reduce_sum(tf.square(L.weight()), axis=[0, 1, 2])).numpy()
    np.testing.assert_allclose(per_unit, L.g_mag.numpy(), atol=1e-4)      # ||w|| == g_mag per filter


def test_conv_enable_requires_realized_wts():
    L = Conv2DPCNLayer(5, (3, 3), 1e-2, "linear", padding="SAME")   # wts not realized
    try:
        L.enable_weight_norm()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
```

- [ ] **Step 2: Run the test to verify it fails.**
Run: `python3 -m pytest tests/test_weight_norm.py -k conv -v`
Expected: FAIL with `AttributeError: 'Conv2DPCNLayer' object has no attribute 'weight'` (or `weight_norm`).

- [ ] **Step 3: Add the `__init__` defaults.** In `Conv2DPCNLayer.__init__`, after the `self.input_shape = None` line, add:
```python
        self.weight_norm = False          # plain Python bool: tf.function branches resolve at trace time
        self.g_mag = None                 # per-output-filter magnitude, created by enable_weight_norm
```

- [ ] **Step 4: Add `weight()` and `enable_weight_norm()`.** Add these two methods to `Conv2DPCNLayer` (e.g. just above `predict_prev`):
```python
    def weight(self):
        # The effective weight used identically in predict_next (encode) and predict_prev
        # (decode). Off = self.wts (byte-identical). On = per-output-filter magnitude g_mag
        # times the unit direction wts/||wts||, normalized over the (kh, kw, in) axes.
        if not self.weight_norm:
            return self.wts
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.wts), axis=[0, 1, 2], keepdims=True)) + 1e-8
        return tf.reshape(self.g_mag, (1, 1, 1, -1)) * self.wts / norm

    def enable_weight_norm(self):
        # Seamless enable: g_mag = per-filter ||wts||, so weight() == wts at enable time.
        if self.wts is None:
            raise RuntimeError("realize weights (run a forward pass) before enabling weight_norm")
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.wts), axis=[0, 1, 2]))   # (O,)
        self.g_mag = tf.Variable(norm, trainable=False)
        self.weight_norm = True
```

- [ ] **Step 5: Run the conv tests to verify they pass.**
Run: `python3 -m pytest tests/test_weight_norm.py -k conv -v`
Expected: PASS (3 conv tests).

- [ ] **Step 6: Route the forward/decode/state paths through `weight()`.** In `Conv2DPCNLayer`, replace the FILTER argument `self.wts` with `self.weight()` at these call sites ONLY. Leave every `self.wts.shape` (sizing) and the `if self.wts is None` guard as raw `self.wts`.
  - `predict_prev`: the two `tf.nn.conv2d_transpose(self.state, self.wts, ...)` calls (stride==1 branch and else branch) become `tf.nn.conv2d_transpose(self.state, self.weight(), ...)`. Keep the `self.wts.shape[-2]` in the `output_shape` tuples unchanged.
  - `pred_loss_d_input`: all four `tf.nn.conv2d_transpose(..., self.wts, strides=..., padding=..., output_shape=x.shape)` become `..., self.weight(), ...`.
  - `update_state`: the four `tf.nn.conv2d(-multiplier*(...), self.wts, strides=self.stride, padding=self.padding)` (relu/gelu/silu/else branches of the `prev_layer` block) become `tf.nn.conv2d(-multiplier*(...), self.weight(), ...)`.
  - `net_in`: `return tf.nn.conv2d(x, self.wts, padding=self.padding, strides=self.stride)` becomes `return tf.nn.conv2d(x, self.weight(), padding=self.padding, strides=self.stride)`. Keep the `if self.wts is None: self.init_params(x.shape)` guard as raw `self.wts`.
  - In `update_wts`, leave the four `filter_sizes=self.wts.shape` unchanged (sizing). The `input=` and `out_backprop=` there already route through `self(...)` and `self.predict_prev()`/`self.predict_next()`, so they pick up `weight()` automatically.

- [ ] **Step 7: Add the weight-norm branch to `update_wts`.** In `Conv2DPCNLayer.update_wts`, replace the LARS block under `if not self.is_clamped or not self.prev_layer.is_clamped:` with a branch. The `else` is byte-identical to the current code:
```python
            if not self.is_clamped or not self.prev_layer.is_clamped:
                denom = (tf.cast(tf.logical_not(self.is_clamped), tf.float32) + tf.cast(tf.logical_not(self.prev_layer.is_clamped), tf.float32))
                g = (d_state + d_pred) / denom
                wd = self.weight_decay
                if self.weight_norm:
                    # Split the local gradient g (w.r.t. the effective weight) into a radial
                    # magnitude update and a tangential direction update, per output filter.
                    # ||w|| = |g_mag| stays bounded (damped by wd); ||wts|| ~preserved (dv ⊥ vhat).
                    norm = tf.sqrt(tf.reduce_sum(tf.square(self.wts), axis=[0, 1, 2], keepdims=True)) + 1e-8
                    vhat = self.wts / norm
                    dg = tf.reduce_sum(g * vhat, axis=[0, 1, 2])                 # (O,)
                    dv = (tf.reshape(self.g_mag, (1, 1, 1, -1)) / norm) * (g - tf.reshape(dg, (1, 1, 1, -1)) * vhat)
                    self.g_mag.assign_sub(self.learning_rate * (dg + wd * self.g_mag))
                    self.wts.assign_sub(self.learning_rate * dv)
                else:
                    wn = tf.norm(self.wts)
                    trust = wn / (tf.norm(g) + wd * wn + 1e-6)
                    trust = tf.minimum(trust, self.trust_cap)
                    self.last_trust = trust  # exposed for logging only
                    self.wts.assign_sub(self.learning_rate * trust * (g + wd * self.wts))
```

- [ ] **Step 8: Add a real-path conv norm-preservation test.** This runs the ACTUAL `update_wts` weight-norm branch end to end (not a re-derivation of the formula) and asserts the direction norm `||wts||` stays put while the magnitude `g_mag` moves. Append to `tests/test_weight_norm.py`:
```python
def test_conv_update_preserves_wts_norm():
    prev = Conv2DPCNLayer(3, (3, 3), 1e-3, "linear", padding="SAME")
    L = Conv2DPCNLayer(5, (3, 3), 1e-3, "linear", padding="SAME", prev_layer=prev)
    x = tf.random.normal((2, 8, 8, 3))
    prev(x, set_state=True)                    # prev.state (2,8,8,3)
    L(prev.predict_next(), set_state=True)     # L.wts (3,3,3,5), L.state (2,8,8,5)
    L.enable_weight_norm()
    n0 = tf.sqrt(tf.reduce_sum(tf.square(L.wts), axis=[0, 1, 2])).numpy()
    g0 = L.g_mag.numpy().copy()
    for _ in range(5):
        L.state.assign(L.state + tf.random.normal(L.state.shape) * 0.1)   # keep the gradient nonzero
        L.update_wts()
    n1 = tf.sqrt(tf.reduce_sum(tf.square(L.wts), axis=[0, 1, 2])).numpy()
    np.testing.assert_allclose(n1, n0, rtol=0.1)          # tangential step keeps ||wts|| put
    assert np.abs(L.g_mag.numpy() - g0).max() > 1e-6      # magnitude actually moved
```
Run: `python3 -m pytest tests/test_weight_norm.py -k conv -v`
Expected: PASS (4 conv tests).

- [ ] **Step 9: Commit.**
```bash
git add conv_pcn_layer.py tests/test_weight_norm.py
git commit -m "gave the conv layer a weight-norm reparameterization: w = g_mag * wts/||wts|| per output filter, fed through a weight() accessor used the same way in predict_next and predict_prev, with update_wts splitting the local gradient into a radial magnitude step and a tangential direction step. off (default) returns self.wts and takes the old LARS path, byte-identical"
```

---

### Task 3: DensePCNLayer weight normalization

**Files:**
- Modify: `dense_pcn_layer.py` (class `DensePCNLayer`)
- Test: `tests/test_weight_norm.py` (dense cases)

**Interfaces:**
- Consumes: nothing from other tasks (mirrors Task 2 for dense shapes).
- Produces:
  - `DensePCNLayer.weight() -> tf.Tensor` (returns `self.wts` when off, else `g_mag * wts / ||wts||` normalized over axis 0).
  - `DensePCNLayer.enable_weight_norm() -> None` (creates `self.g_mag` = per-output `||wts||` over axis 0, sets `self.weight_norm = True`; raises if `wts` is None).
  - `self.weight_norm: bool` (default False), `self.g_mag: tf.Variable | None` (default None).

- [ ] **Step 1: Write the failing test.** Append dense cases to `tests/test_weight_norm.py`:
```python
def _realize_dense():
    L = DensePCNLayer(3, 1e-2, "linear")
    L(tf.random.normal((8, 4)), set_state=True)   # realizes wts (4,3) and b (3,)
    return L


def test_dense_weight_off_is_wts_identity():
    L = _realize_dense()
    assert L.weight_norm is False
    assert L.weight() is L.wts


def test_dense_enable_is_seamless():
    L = _realize_dense()
    w_before = L.weight().numpy().copy()
    L.enable_weight_norm()
    assert L.weight_norm is True
    np.testing.assert_allclose(L.weight().numpy(), w_before, atol=1e-5)
    per_unit = tf.norm(L.weight(), axis=0).numpy()            # per output column
    np.testing.assert_allclose(per_unit, L.g_mag.numpy(), atol=1e-4)


def test_dense_update_preserves_wts_norm():
    # Runs the real update_wts weight-norm branch: ||wts|| per column stays put
    # (tangential direction step) while g_mag moves (radial magnitude step).
    prev = DensePCNLayer(4, 1e-3, "linear")
    L = DensePCNLayer(3, 1e-3, "linear", prev_layer=prev)
    x = tf.random.normal((8, 4))
    prev(x, set_state=True)                  # prev.state (8,4)
    L(prev.predict_next(), set_state=True)   # L.wts (4,3), L.b (3,), L.state (8,3)
    L.enable_weight_norm()
    n0 = tf.norm(L.wts, axis=0).numpy()
    g0 = L.g_mag.numpy().copy()
    for _ in range(5):
        L.state.assign(L.state + tf.random.normal(L.state.shape) * 0.1)
        L.update_wts()
    n1 = tf.norm(L.wts, axis=0).numpy()
    np.testing.assert_allclose(n1, n0, rtol=0.1)
    assert np.abs(L.g_mag.numpy() - g0).max() > 1e-6
```

- [ ] **Step 2: Run the test to verify it fails.**
Run: `python3 -m pytest tests/test_weight_norm.py -k dense -v`
Expected: FAIL with `AttributeError: 'DensePCNLayer' object has no attribute 'weight'`.

- [ ] **Step 3: Add the `__init__` defaults.** In `DensePCNLayer.__init__`, after `self.share_state_layer = share_state_layer`, add:
```python
        self.weight_norm = False          # plain Python bool
        self.g_mag = None                 # per-output-unit magnitude, created by enable_weight_norm
```

- [ ] **Step 4: Add `weight()` and `enable_weight_norm()`.** Add to `DensePCNLayer` (e.g. just above `predict_prev`):
```python
    def weight(self):
        # Effective weight, used identically in predict_next (encode) and predict_prev (decode).
        # Off = self.wts. On = per-output-column magnitude g_mag times wts/||wts||, normalized
        # over axis 0 (the input dim); wts is (in, out), g_mag is (out,).
        if not self.weight_norm:
            return self.wts
        norm = tf.norm(self.wts, axis=0, keepdims=True) + 1e-8      # (1, out)
        return self.g_mag[None, :] * self.wts / norm

    def enable_weight_norm(self):
        if self.wts is None:
            raise RuntimeError("realize weights (run a forward pass) before enabling weight_norm")
        norm = tf.norm(self.wts, axis=0)     # (out,)
        self.g_mag = tf.Variable(norm, trainable=False)
        self.weight_norm = True
```

- [ ] **Step 5: Run the dense tests to verify they pass.**
Run: `python3 -m pytest tests/test_weight_norm.py -k dense -v`
Expected: PASS (3 dense tests).

- [ ] **Step 6: Route the forward/decode/state/bias paths through `weight()`.** In `DensePCNLayer`, replace `self.wts` with `self.weight()` at these operand sites ONLY. Leave the `if self.wts is None` guard, the `tf.norm(self.wts)`/`assign_sub` inside the LARS branch, and any `self.wts.shape` as raw `self.wts`.
  - `predict_prev`: `(self.state - self.b) @ tf.linalg.matrix_transpose(self.wts)` becomes `... matrix_transpose(self.weight())`.
  - `pred_loss_d_input`: both `@ tf.linalg.matrix_transpose(self.wts)` become `@ tf.linalg.matrix_transpose(self.weight())`.
  - `update_state`: the two `@ self.wts` in the `prev_layer` d_pred block (relu and else) become `@ self.weight()`.
  - `net_in`: `return x @ self.wts + self.b` becomes `return x @ self.weight() + self.b`. Keep the `if self.wts is None: self.init_params(x.shape)` guard as raw `self.wts`.
  - `update_b`: the two `@ self.wts` (relu and else branches of the `not self.is_clamped` block) become `@ self.weight()`. This keeps the bias gradient consistent with the effective decode weight; it is byte-identical when weight_norm is off (`weight() is self.wts`). NOTE: this refines the spec line that said "update_b unchanged"; the correct bias gradient flows through `weight()`, and the change is inert when off.
  - In `update_wts`, the d_state/d_pred computation already routes through `self(...)`, `self.net_in(...)`, and `self.predict_prev()`, so it picks up `weight()` automatically. Do not add direct `self.wts` there.

- [ ] **Step 7: Add the weight-norm branch to `update_wts`.** In `DensePCNLayer.update_wts`, replace the LARS block under `if not self.is_clamped or not self.prev_layer.is_clamped:` with a branch (the `else` is byte-identical to the current code):
```python
            if not self.is_clamped or not self.prev_layer.is_clamped:
                denom = tf.cast(int(not self.is_clamped)+int(not self.prev_layer.is_clamped), tf.float32)
                g = (d_state + d_pred) / denom
                wd = self.weight_decay
                if self.weight_norm:
                    norm = tf.norm(self.wts, axis=0, keepdims=True) + 1e-8       # (1, out)
                    vhat = self.wts / norm
                    dg = tf.reduce_sum(g * vhat, axis=0)                         # (out,)
                    dv = (self.g_mag[None, :] / norm) * (g - dg[None, :] * vhat)
                    self.g_mag.assign_sub(self.learning_rate * (dg + wd * self.g_mag))
                    self.wts.assign_sub(self.learning_rate * dv)
                else:
                    wn = tf.norm(self.wts)
                    trust = wn / (tf.norm(g) + wd * wn + 1e-6)
                    trust = tf.minimum(trust, self.trust_cap)
                    self.last_trust = trust  # exposed for logging only
                    self.wts.assign_sub(self.learning_rate * trust * (g + wd * self.wts))
```

- [ ] **Step 8: Run the full weight-norm suite.**
Run: `python3 -m pytest tests/test_weight_norm.py -v`
Expected: PASS (all conv and dense cases).

- [ ] **Step 9: Commit.**
```bash
git add dense_pcn_layer.py tests/test_weight_norm.py
git commit -m "gave the dense layer the same weight-norm reparameterization as the conv layer: per-output-column magnitude g_mag times wts/||wts||, a weight() accessor used both directions, update_wts split into a radial magnitude step and a tangential direction step, and the bias gradient routed through weight() too. off returns self.wts and takes the old LARS path, byte-identical"
```

---

### Task 4: Inertness gate (COCO64 now, NATIVE-143 deferred)

Proves the reparameterization is byte-identical when weight-norm is off, by regenerating the COCO64 signature from the post-change code and comparing to the reference banked in Task 1.

**Files:**
- Uses: `tools/rewrite_gate.py`, `tools/gate_compare.py`, `tools/clusterrun.sh`, `docs/superpowers/gate_ref_coco64.npz`

**Interfaces:**
- Consumes: `docs/superpowers/gate_ref_coco64.npz` (Task 1), the changed `conv_pcn_layer.py` and `dense_pcn_layer.py` (Tasks 2, 3).
- Produces: a `GATE_MATCH` verdict recorded in `docs/experiments/LOG.md`.

- [ ] **Step 1: Regenerate the COCO64 signature from the changed code and compare.** The changed layers run with `weight_norm` defaulting off (no `--weight-norm`, no `enable_weight_norm`), so the signature must match the reference.
```
tools/clusterrun.sh --name gatechk --gpu L4 --mem 40G --cpus 4 --time 00:20:00 \
  --sync "rewrite_gate.py gate_compare.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py docs/superpowers/gate_ref_coco64.npz" \
  --run "python3 tools/rewrite_gate.py --config coco64 --relaxed --relax-steps 5 --weight-steps 2 --save /tmp/gate_cur_coco64.npz && python3 tools/gate_compare.py docs/superpowers/gate_ref_coco64.npz /tmp/gate_cur_coco64.npz 1e-4"
```
Expected: `GATE_MATCH nlayers=<N> tol=0.0001` (same N as Task 1). If `GATE_MISMATCH`, a `weight()` substitution changed numerics when off; inspect the mismatched layer index, fix the offending site (a wrong `self.wts` -> `self.weight()` in a sizing spot, or a missed guard), and re-run.

- [ ] **Step 2: Record the deferred NATIVE-143 gate.** The canonical `GATE_MATCH nlayers=143` for NATIVE_7B needs a 40GB+ GPU (H200/n15 drained). Add a NOTE to `docs/STATE.md` that the NATIVE-143 gate is pending a big-GPU window, with the interim proof being (a) the COCO64 GATE_MATCH above, (b) the provable inertness (weight_norm defaults off, `weight()` returns `self.wts`, `update_wts` else-branch is the unchanged LARS), and (c) the unit tests. When a 40GB+ GPU frees, first locate the banked NATIVE golden signature (search the repo and `~/encoder_encoder_PCN` for the committed `*.npz` reference used by the last `GATE_MATCH nlayers=143`; if none survives, regenerate it from the pre-change code at the last-good commit before Task 2). Then run the two-step gate: generate the current signature from the changed code with the flag off and compare.
```
tools/clusterrun.sh --name gate143 --gpu H200 --mem 120G --cpus 8 --time 00:40:00 \
  --sync "rewrite_gate.py gate_compare.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py NATIVE_REF.npz" \
  --run "python3 tools/rewrite_gate.py --config native7b --relaxed --save /tmp/native_cur.npz && python3 tools/gate_compare.py NATIVE_REF.npz /tmp/native_cur.npz 1e-4"
```
where `NATIVE_REF.npz` is the located/regenerated banked signature. Expect `GATE_MATCH nlayers=143`.

- [ ] **Step 3: Log the gate result.** Prepend a dated entry to `docs/experiments/LOG.md` recording the COCO64 GATE_MATCH (with N) and the NATIVE-143 deferral. Update `docs/STATE.md`. Commit:
```bash
git add docs/experiments/LOG.md docs/STATE.md
git commit -m "gated the weight-norm change: COCO64 signature is byte-identical with the flag off (GATE_MATCH), NATIVE-143 gate deferred to a big-GPU window with provable inertness plus the unit tests as the interim proof"
```

---

### Task 5: `--weight-norm` flag, enable-after-restore, and g_mag persistence

**Files:**
- Modify: `train_coco64.py` (`main`)
- Test: (cluster) 4-epoch plumbing smoke on COCO64_GEN

**Interfaces:**
- Consumes: `Conv2DPCNLayer.enable_weight_norm` / `DensePCNLayer.enable_weight_norm` (Tasks 2, 3), the model's `trainable_layers`.
- Produces: `--weight-norm` (store_true, default off); when set, every conv/dense trainable layer with realized `wts` is weight-normed AFTER any resume-restore, and each layer's `g_mag` is checkpointed to `<ckpt>_wn` / `<ckpt>_best_wn` so a resume or retest restores the trained magnitude. Off is unchanged.

- [ ] **Step 1: Add the flag.** In `main`, next to the other `add_argument` calls (after `--gen-lr`), add:
```python
    ap.add_argument("--weight-norm", action="store_true")   # PC-native weight-norm stabilizer on conv/dense layers
```

- [ ] **Step 2: Enable weight-norm after the resume-restore, and set up g_mag persistence.** The enable MUST come after the `if a.resume and mgr.latest_checkpoint: ckpt.restore(...)` line so `g_mag` derives from the warm-started (restored) weights and stays seamless. Immediately after that `if a.resume ...` block (the line `ckpt.restore(mgr.latest_checkpoint); print("resumed", ...)`), add:
```python
    WN_W = []
    wn_mgr = wn_best_mgr = None
    if a.weight_norm:
        nwn = 0
        for L in m.trainable_layers:
            if hasattr(L, "enable_weight_norm") and getattr(L, "wts", None) is not None:
                L.enable_weight_norm(); nwn += 1
        WN_W = [L.g_mag for L in m.trainable_layers if getattr(L, "weight_norm", False)]
        print(f"weight_norm enabled on {nwn} conv/dense layers", flush=True)
        if WN_W:
            wn_ckpt = tf.train.Checkpoint(**{f"g{i}": v for i, v in enumerate(WN_W)})
            wn_mgr = tf.train.CheckpointManager(wn_ckpt, a.ckpt + "_wn", max_to_keep=1)
            wn_best_mgr = tf.train.CheckpointManager(wn_ckpt, a.ckpt + "_best_wn", max_to_keep=1)
            if a.resume and wn_mgr.latest_checkpoint:
                wn_ckpt.restore(wn_mgr.latest_checkpoint)   # restore trained magnitudes over the ||wts||-derived ones
                print("resumed wn", wn_mgr.latest_checkpoint, flush=True)
```
NOTE: the base checkpoint (`ALL_W`) is unchanged, so recon checkpoints (no `g_mag`) stay loadable and the positional `v{i}` scheme is untouched. When warm-starting from a recon ckpt there is no `<ckpt>_wn`, so `g_mag` stays at the restored `||wts||` (seamless). A partial weight-norm run that is resumed WITHOUT a saved `<ckpt>_wn` would recompute `g_mag = ||wts||` and lose the trained magnitude; the `_wn`/`_best_wn` managers below prevent that.

- [ ] **Step 3: Mirror g_mag into the save sites.** Define two helpers right after the persistence block above:
```python
    def save_latest():
        mgr.save()
        if wn_mgr is not None: wn_mgr.save()
    def save_best():
        best_mgr.save()
        if wn_best_mgr is not None: wn_best_mgr.save()
```
Then in the training loop replace: the best-save `best_e = e; best_mgr.save(); print(...)` uses `save_best()` in place of `best_mgr.save()`; the diverged `mgr.save(); return` uses `save_latest()`; the periodic `mgr.save(); print(f"ckpt @ {step}", ...)` uses `save_latest()`; and the final `mgr.save(); print("TRAIN_DONE", ...)` uses `save_latest()`. When `--weight-norm` is off, `wn_mgr`/`wn_best_mgr` are None so these call only the base managers, unchanged.

- [ ] **Step 4: Syntax-check.**
Run: `python3 -c "import ast; ast.parse(open('train_coco64.py').read())"`
Expected: no output (parses clean).

- [ ] **Step 5: Plumbing smoke on the cluster (L4).** Confirms `--weight-norm` enables, `g_mag` is created, the weight-norm branch runs in both recon and CHL steps, clamp hygiene holds, and states stay finite. NOT a quality or long-stability check.
```
tools/clusterrun.sh --name wn_smoke --gpu L4 --mem 40G --cpus 4 --time 00:25:00 \
  --sync "train_coco64.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py coco64_data.py infonce.py" \
  --run "python3 train_coco64.py --config coco64_gen --weight-norm --train-mode chl --gen-lr 3e-4 --pairs 64 --epochs 4 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --energy-every 5 --ckpt ckpt_wn_smoke"
```
Expect: `config=coco64_gen`, a `weight_norm enabled on <n> conv/dense layers` line with n > 0, finite energy lines, `max_abs_state` under 400, NO `DIVERGED`/NaN, NO clamp-signature `RuntimeError`, and `TRAIN_DONE`. Leave `ckpt_wn_smoke*` unstaged.

- [ ] **Step 6: Commit.**
```bash
git add train_coco64.py
git commit -m "added --weight-norm: after any resume-restore it enables the weight-norm reparameterization on every conv/dense layer (g_mag seeded from the warm-started ||wts|| so it is seamless), and it checkpoints each g_mag to <ckpt>_wn / <ckpt>_best_wn so a resume or retest restores the trained magnitude. off is unchanged"
```

---

### Task 6: CHL retrain with weight-norm and the make-or-break stability gate

The core experiment. Warm-start from the recon-trained COCO64_GEN best, enable weight-norm, run CHL, and verify training now holds WELL PAST ep13 without the state pinning at the clip. This is a long detached run watched with a corrected failure detector.

**Files:**
- Create: `tools/run_chl_wn.sh` (sbatch script)
- Uses: the recon best checkpoint (the COCO64_GEN reconstruction ckpt used to warm-start prior CHL runs), `train_coco64.py`

**Interfaces:**
- Consumes: `--weight-norm` + `--train-mode chl` (Task 5), a warm-start seed.
- Produces: `ckpt_chl_wn` (+ `_best`, `_wn`, `_best_wn`) and a stability verdict logged to `docs/experiments/LOG.md`.

- [ ] **Step 1: Seed the warm start.** On the cluster, copy the recon-trained COCO64_GEN best checkpoint directory `ckpt_gen_best` (the clean recon best used to warm-start prior CHL runs, per `docs/experiments/LOG.md`) to `ckpt_chl_wn`. Confirm the exact name against `docs/STATE.md` first in case it moved. This lets `--resume` restore the recon weights, after which weight-norm enable seeds `g_mag = ||recon wts||` (seamless). If `ckpt_chl_wn` already exists from a prior attempt, remove it first so the seed is clean.

- [ ] **Step 2: Write the sbatch script `tools/run_chl_wn.sh`.**
```bash
#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:L4:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -J p8_chlwn
#SBATCH -o chlwn_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
python3 train_coco64.py --config coco64_gen --weight-norm --train-mode chl --gen-lr 3e-4 --gen-every 4 --pairs 2000 --epochs 30 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --energy-every 50 --ckpt ckpt_chl_wn --resume
```

- [ ] **Step 3: Sync the code and launch detached.** Sync the changed files to the cluster (via a `clusterrun.sh` no-op sync, or `scp`), then submit `sbatch tools/run_chl_wn.sh` over ssh (`ssh slsang29@hpc.colby.edu`, requires the Colby VPN). Record the job id.

- [ ] **Step 4: Watch with the CORRECTED failure detector.** Poll the job log with the Monitor tool. The earlier CHL run reported a clean `TRAIN_DONE` on a blown-up run because the grep only watched `DIVERGED`/`RuntimeError`. Watch ALSO for the norm-inflation signature: parse `max_abs_state=` from each `[step ...]` line and flag when it approaches the clip (e.g. `> 350`, clip is 400) or when `energy=` climbs across consecutive prints. The make-or-break question is whether `max_abs_state` stays bounded and does NOT climb to the clip PAST ep13 (the wall every prior objective hit). Success = stable well past ep13 (roughly step > 3250 at 2000 pairs / batch 8 / gen-every 4). If it destabilizes at ~ep13 like the un-normalized runs, the weight-norm did not stabilize; capture the last stable `_best` ckpt and escalate to the fix-`g_mag`-at-init fallback (learn direction only) from the spec.

- [ ] **Step 5: Log the stability outcome.** Prepend a dated entry to `docs/experiments/LOG.md`: did training hold past ep13, the max `max_abs_state`, the final energy, and how far it ran. Update `docs/STATE.md`. Commit:
```bash
git add tools/run_chl_wn.sh docs/experiments/LOG.md docs/STATE.md
git commit -m "ran the CHL objective with weight-norm warm-started from the recon best; recorded whether the reparameterization holds training past the ep13 norm-inflation wall that capped every prior generative run"
```

---

### Task 7: Darkness and text-to-image retest on the weight-norm best checkpoint

Interprets the experiment: with training holding longer, does the decode brighten and sharpen toward caption-specific structure, with recon and image-to-caption intact.

**Files:**
- Modify: `tools/darkness_diag.py`, `tools/gen_retest.py` (add a `--weight-norm` path that enables weight-norm on the layers and restores `g_mag` from `<ckpt>_best_wn`)
- Uses: `ckpt_chl_wn_best` + `ckpt_chl_wn_best_wn`

**Interfaces:**
- Consumes: the best weight-norm checkpoint from Task 6, the retest tools.
- Produces: brightness/contrast ratios and per-caption top-down generation quality, logged to `docs/experiments/LOG.md`.

- [ ] **Step 1: Add a `--weight-norm` restore path to the retest tools.** In `tools/darkness_diag.py` and `tools/gen_retest.py`, after the model is built and the base checkpoint is restored, add: if a `--weight-norm` flag is passed, enable weight-norm on every conv/dense layer (`for L in m.trainable_layers: if hasattr(L, "enable_weight_norm") and getattr(L, "wts", None) is not None: L.enable_weight_norm()`), then restore `g_mag` from `<ckpt>_best_wn` via a `tf.train.Checkpoint` over `[L.g_mag for L in m.trainable_layers if getattr(L, "weight_norm", False)]` in the SAME `trainable_layers` order used in Task 5 (so the `g{i}` keys line up). Without this, the retest would recompute `g_mag = ||wts||` and not reproduce the trained decode. Keep the non-`--weight-norm` path unchanged.

- [ ] **Step 2: Run the darkness diagnostic on the cluster (L4).** Point it at `ckpt_chl_wn_best` with `--weight-norm`. Report the text-set vs image-set latent scale and the generated-image mean/std/min/max vs the true image (the darkness/contrast ratios), comparing to the prior CHL best (std ratio 0.40, mean ratio 0.38).

- [ ] **Step 3: Run the text-to-image retest on the cluster (L4).** Point `gen_retest.py` (and, if useful, the top-down-boost sweep) at `ckpt_chl_wn_best` with `--weight-norm`. Report the boosted-generation contrast/brightness and whether per-caption structure is more recognizable than the prior blob.

- [ ] **Step 4: Log the retest and update STATE.** Prepend a dated entry to `docs/experiments/LOG.md` with the ratios and the qualitative verdict (did weight-norm plus a longer CHL horizon brighten and sharpen text-to-image, or did the ceiling hold). Update `docs/STATE.md` with the new status and the next lever. Commit:
```bash
git add tools/darkness_diag.py tools/gen_retest.py docs/experiments/LOG.md docs/STATE.md
git commit -m "retested text-to-image on the weight-norm CHL best: added a --weight-norm restore path to the diagnostics and recorded the brightness, contrast, and per-caption sharpness versus the prior CHL blob"
```

---

## Notes for the implementer

- The whole change is clamp + the model's own `update_state` relaxation + the existing local `update_wts`/`update_b`, reparameterized. NO backprop, NO separate decoder, NO optimizer. `weight()` is the ONLY new thing in the forward/decode path and it is the SAME tensor used both directions. This is the core project constraint (bidirectional PC); do not deviate.
- `weight_norm` is a plain Python bool so `tf.function`-compiled relaxation resolves the branch at trace time; enable it BEFORE the first traced training step (Task 5 does, right after restore). Never make it a `tf.Variable`.
- Off (default, no `--weight-norm`) must be byte-identical: `weight()` returns `self.wts`, `update_wts` takes the unchanged LARS `else`, and the base checkpoint is untouched. The Task 4 COCO64 GATE_MATCH is the guard; the NATIVE-143 gate is the canonical proof, deferred to a big-GPU window.
- Do not add `g_mag` to the base `ALL_W` checkpoint. It rides in the separate `_wn`/`_best_wn` managers so recon checkpoints stay loadable and the positional `v{i}` scheme is preserved.
- The CHL step (`chl_step`) needs NO change: it calls `update_wts`, which now branches internally. With `weight_decay=0` during the CHL contrast steps, `g_mag` is not damped there; its damping comes from the interleaved recon steps (`weight_decay` on).
