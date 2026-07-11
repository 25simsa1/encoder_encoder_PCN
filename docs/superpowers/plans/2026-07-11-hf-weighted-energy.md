# High-frequency-weighted PC energy (Approach A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Boost the high-frequency component of the bottom image prediction error so the decode's local weight step learns sharp detail instead of the blurry conditional mean, and test on the 2k overfit whether generation sharpens.

**Architecture:** In `conv1.update_wts` (the layer whose `prev_layer` is `img_input`, so its d_pred term is the pixel-space error), reweight the error `e` to `e + hf_gamma * Laplacian(e)` via a fixed depthwise high-pass filter. `hf_gamma = 0` is `e` exactly (byte-identical). Opt-in via `--hf-weight`, set on the bottom conv only, composes with `--weight-norm`.

**Tech Stack:** Python, TensorFlow 2.21 (eager + the compiled relaxation sweep), pytest, the Colby HPC L4 nodes via `tools/clusterrun.sh`.

## Global Constraints

- **Bidirectional PC only.** The high-pass is a FIXED linear filter on the layer's own local prediction error, so the update stays a local rule. NO backprop, NO separate decoder, NO optimizer.
- **NATIVE stays byte-identical.** `hf_gamma` defaults 0.0; the `_hf` branch returns `e` unchanged when off, so `update_wts` is byte-identical. The golden gate `GATE_MATCH nlayers=143` must still pass. `hf_gamma` is a plain Python float (so the compiled sweep's branch resolves at trace time).
- **OOM is expected; do not shrink the model.** Use L4 (n7/n8) for COCO64_GEN. H200/n15 drained. n10 A100 is another project's run.
- **The decisive bar does not move** (8k, >3/2000). Not touched here.
- **Commit style:** first-person student voice, no AI attribution, no `Co-Authored-By`, no "Generated with". Identity `Simon Sang <simonlapsang@gmail.com>`. Commit locally, push at checkpoints.
- **Writing style:** no em dashes, no colons, in any prose or docs authored here.
- **Cluster caveat:** `tools/clusterrun.sh` syncs repo-relative paths (use the `tools/` prefix for tools), fetches only with `--fetch`, and cannot take an inline `python3 -c` with single quotes. A lone L4 `GATE_MISMATCH` at ~1e-4 is cross-node cuDNN fp noise; re-check same-node before calling it a regression.
- **Docs protocol:** after any run or work chunk, prepend a dated entry to `docs/experiments/LOG.md` (never edit past entries) and update `docs/STATE.md`.

---

### Task 1: Conv2DPCNLayer high-frequency weighting

**Files:**
- Modify: `conv_pcn_layer.py` (class `Conv2DPCNLayer`)
- Test: `tests/test_hf_weight.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Conv2DPCNLayer.hf_gamma: float` (default 0.0).
  - `Conv2DPCNLayer._hf(e) -> tf.Tensor` returns `e` when `hf_gamma == 0.0`, else `e + hf_gamma * depthwise_Laplacian(e)` (per-channel high-pass, SAME padding, shape preserved).
  - `update_wts`'s d_pred error wrapped in `self._hf(...)` in all four activation branches.

- [ ] **Step 1: Write the failing test.** Create `tests/test_hf_weight.py`:
```python
import numpy as np
import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer


def test_hf_off_returns_error_unchanged():
    L = Conv2DPCNLayer(3, (3, 3), 1e-3, "linear", padding="SAME")
    assert L.hf_gamma == 0.0
    e = tf.random.normal((2, 8, 8, 3))
    np.testing.assert_array_equal(L._hf(e).numpy(), e.numpy())   # off = identity, byte-identical


def test_hf_leaves_smooth_error_but_boosts_edges():
    L = Conv2DPCNLayer(3, (3, 3), 1e-3, "linear", padding="SAME")
    L.hf_gamma = 1.0
    smooth = tf.ones((1, 8, 8, 3))                                # constant -> Laplacian ~ 0
    np.testing.assert_allclose(L._hf(smooth).numpy(), smooth.numpy(), atol=1e-4)
    sharp = np.zeros((1, 8, 8, 3), np.float32); sharp[0, 4, 4, :] = 1.0   # impulse -> big Laplacian
    out = L._hf(tf.constant(sharp)).numpy()
    assert np.abs(out - sharp).max() > 0.5                        # the edge is boosted
```

- [ ] **Step 2: Run the test to verify it fails.**
Run: `python3 -m pytest tests/test_hf_weight.py -v`
Expected: FAIL with `AttributeError: 'Conv2DPCNLayer' object has no attribute 'hf_gamma'` (or `_hf`).

- [ ] **Step 3: Add the `__init__` default.** In `Conv2DPCNLayer.__init__`, after the `self.g_mag = None` line (added by the weight-norm work), add:
```python
        self.hf_gamma = 0.0               # high-frequency boost on the bottom pixel error; 0 = off
```

- [ ] **Step 4: Add the `_hf` method.** Add to `Conv2DPCNLayer` (e.g. just above `update_wts`):
```python
    def _hf(self, e):
        # High-frequency boost of a bottom prediction error: e + hf_gamma * Laplacian(e), a
        # fixed depthwise high-pass so this stays the layer's own local error (no backprop).
        # hf_gamma == 0 returns e unchanged (byte-identical). Built fresh each call (a tiny
        # graph constant), so it is safe inside the compiled relaxation sweep.
        if self.hf_gamma == 0.0:
            return e
        c = int(e.shape[-1])
        lap = tf.constant([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], tf.float32)
        kernel = tf.reshape(lap, (3, 3, 1, 1)) * tf.ones((1, 1, c, 1), tf.float32)
        hp = tf.nn.depthwise_conv2d(e, kernel, strides=[1, 1, 1, 1], padding="SAME")
        return e + self.hf_gamma * hp
```

- [ ] **Step 5: Wrap the d_pred error in `update_wts`.** In `Conv2DPCNLayer.update_wts`, inside the `if not self.is_clamped:` block, wrap the `input=` error of EACH `Conv2DBackpropFilter` d_pred branch with `self._hf(...)`. The four branches become:
```python
                if self.activation == 'relu':
                    d_pred += tf.raw_ops.Conv2DBackpropFilter(input=self._hf(tf.nn.relu(pred)-tf.nn.relu(self.prev_layer.predict_next())), filter_sizes=self.wts.shape, out_backprop=self.predict_next(), strides=[1, self.stride, self.stride, 1], padding=self.padding)
                elif self.activation == 'gelu':
                    d_pred += tf.raw_ops.Conv2DBackpropFilter(input=self._hf(tf.nn.gelu(pred)-tf.nn.gelu(self.prev_layer.predict_next())), filter_sizes=self.wts.shape, out_backprop=self.predict_next(), strides=[1, self.stride, self.stride, 1], padding=self.padding)
                elif self.activation == 'silu':
                    d_pred += tf.raw_ops.Conv2DBackpropFilter(input=self._hf(tf.nn.silu(pred)-tf.nn.silu(self.prev_layer.predict_next())), filter_sizes=self.wts.shape, out_backprop=self.predict_next(), strides=[1, self.stride, self.stride, 1], padding=self.padding)
                else:
                    d_pred += tf.raw_ops.Conv2DBackpropFilter(input=self._hf(pred-self.prev_layer.predict_next()), filter_sizes=self.wts.shape, out_backprop=self.predict_next(), strides=[1, self.stride, self.stride, 1], padding=self.padding)
```
Do NOT touch the d_state branch (the `if not self.prev_layer.is_clamped:` block above it) or the weight-step branch below. Only the four d_pred `input=` args gain the `self._hf(...)` wrapper. When `hf_gamma == 0` (every conv except the bottom one, and all convs when the flag is off), `_hf` returns the error unchanged, so the op tree is byte-identical.

- [ ] **Step 6: Run the tests to verify they pass.**
Run: `python3 -m pytest tests/test_hf_weight.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Add a real-path integration test.** Append to `tests/test_hf_weight.py`, confirming `update_wts` produces a DIFFERENT weight delta with the boost on (and identical with it off):
```python
def _pair(hf):
    prev = Conv2DPCNLayer(3, (3, 3), 1e-2, "linear", padding="SAME")
    L = Conv2DPCNLayer(5, (3, 3), 1e-2, "linear", padding="SAME", prev_layer=prev)
    x = tf.random.normal((2, 8, 8, 3), seed=0)
    prev(x, set_state=True)
    L(prev.predict_next(), set_state=True)
    L.hf_gamma = hf
    return L


def test_hf_changes_the_weight_step():
    tf.random.set_seed(0); off = _pair(0.0); w0 = off.wts.numpy().copy(); off.update_wts(); d_off = off.wts.numpy() - w0
    tf.random.set_seed(0); on = _pair(2.0);  w1 = on.wts.numpy().copy();  on.update_wts();  d_on = on.wts.numpy() - w1
    assert np.abs(d_off).max() > 0                       # a real update happened
    assert np.abs(d_on - d_off).max() > 1e-6             # the boost changed it
```
Run: `python3 -m pytest tests/test_hf_weight.py -v`
Expected: PASS (3 tests).

- [ ] **Step 8: Commit.**
```bash
git add conv_pcn_layer.py tests/test_hf_weight.py
git commit -m "gave the conv layer a high-frequency boost on the bottom pixel error: update_wts d_pred error becomes e + hf_gamma*Laplacian(e) via a fixed depthwise high-pass, so the decode weight step learns sharp detail. off (hf_gamma=0, default) returns e unchanged, byte-identical. pure local PC rule"
```

---

### Task 2: COCO64 inertness gate (flag off byte-identical)

Proves the HF change is byte-identical when off, reusing the banked reference from the weight-norm work.

**Files:**
- Uses: `tools/rewrite_gate.py`, `tools/gate_compare.py`, `tools/clusterrun.sh`, `docs/superpowers/gate_ref_coco64.npz` (banked, nlayers=88).

**Interfaces:**
- Consumes: the changed `conv_pcn_layer.py` (Task 1), the banked `gate_ref_coco64.npz`.
- Produces: a `GATE_MATCH` verdict logged to `docs/experiments/LOG.md`.

- [ ] **Step 1: Regenerate the COCO64 signature and compare.** With no `--hf-weight` flag and `hf_gamma` defaulting 0, `rewrite_gate.py` exercises `conv1.update_wts` with `_hf` returning `e` unchanged, so the signature must match the banked reference.
```
tools/clusterrun.sh --name hfgate --gpu L4 --mem 40G --cpus 4 --time 00:20:00 \
  --sync "tools/rewrite_gate.py tools/gate_compare.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py docs/superpowers/gate_ref_coco64.npz" \
  --run "python3 tools/rewrite_gate.py --config coco64 --relaxed --relax-steps 5 --weight-steps 2 --save hf_cur.npz && python3 tools/gate_compare.py docs/superpowers/gate_ref_coco64.npz hf_cur.npz 1e-4"
```
Expected: `GATE_MATCH nlayers=88 tol=0.0001`. If `GATE_MISMATCH` at ~1e-4 on a few layers, it is the cross-node cuDNN fp caveat, re-run once (same-node run-to-run should match); a larger or systematic mismatch means an `_hf` wrapper leaked into a non-off path, fix it.

- [ ] **Step 2: Log the gate.** Prepend a dated entry to `docs/experiments/LOG.md` recording the COCO64 GATE_MATCH (HF off byte-identical) and note the NATIVE-143 gate remains deferred to a big-GPU window (H200 drained), with the COCO64 gate + provable inertness (`_hf` returns `e` when `hf_gamma==0`) + unit tests as the interim proof. Update `docs/STATE.md`. Commit:
```bash
git add docs/experiments/LOG.md docs/STATE.md
git commit -m "gated the HF-weight change: COCO64 signature byte-identical with the flag off (GATE_MATCH nlayers=88), so NATIVE stays safe. NATIVE-143 gate deferred to a big-GPU window"
```

---

### Task 3: `--hf-weight` flag on the bottom conv

**Files:**
- Modify: `train_coco64.py` (`main`)
- Test: (cluster) plumbing smoke on COCO64_GEN

**Interfaces:**
- Consumes: `Conv2DPCNLayer.hf_gamma` (Task 1), `m.img_input`.
- Produces: `--hf-weight GAMMA` (float, default 0.0); when > 0 it sets `hf_gamma` on the bottom conv (the one whose `prev_layer is m.img_input`) only. Off = unchanged.

- [ ] **Step 1: Add the flag.** Next to the other `add_argument` calls (after `--weight-norm`), add:
```python
    ap.add_argument("--hf-weight", type=float, default=0.0)   # high-frequency boost on the bottom pixel error
```

- [ ] **Step 2: Set hf_gamma on the bottom conv.** After the flag-application loops (near where `--conv-activation` / `--weight-decay` are applied, before the weight-realizing `pass_through`), add:
```python
    if a.hf_weight > 0:
        nhf = 0
        for L in m.trainable_layers:
            if isinstance(L, Conv2DPCNLayer) and getattr(L, "prev_layer", None) is m.img_input:
                L.hf_gamma = a.hf_weight; nhf += 1
        print(f"hf_weight={a.hf_weight} set on {nhf} bottom conv layer(s)", flush=True)
```
`hf_gamma` is set before the first compiled sweep, so the trace bakes the boost; off (`--hf-weight 0`) never enters the branch, so it is unchanged. `nhf` should be 1 (only `conv1` has `prev_layer is img_input`).

- [ ] **Step 3: Syntax-check.**
Run: `python3 -c "import ast; ast.parse(open('train_coco64.py').read())"`
Expected: no output.

- [ ] **Step 4: Plumbing smoke (CONTROLLER runs on the cluster).** Confirms `--hf-weight` sets the bottom conv, training stays finite, and it composes with weight-norm. The implementer SKIPS this (no cluster/GPU commands); the controller runs:
```
tools/clusterrun.sh --name hf_smoke --gpu L4 --mem 40G --cpus 4 --time 00:25:00 \
  --sync "train_coco64.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py coco64_data.py infonce.py" \
  --run "python3 train_coco64.py --config coco64_gen --hf-weight 1.0 --weight-norm --train-mode recon --pairs 64 --epochs 3 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --energy-every 5 --ckpt ckpt_hf_smoke"
```
Expect: `config=coco64_gen`, an `hf_weight=1.0 set on 1 bottom conv layer(s)` line, finite energy, `max_abs_state` under 400, no `DIVERGED`/NaN, `TRAIN_DONE`.

- [ ] **Step 5: Commit.**
```bash
git add train_coco64.py
git commit -m "added --hf-weight: sets the high-frequency boost hf_gamma on the bottom conv (the one whose prev_layer is img_input) so the decode weight step is pushed toward sharp detail. off (default 0) is unchanged; composes with --weight-norm"
```

---

### Task 4: The decisive retrain and gamma sweep retest

Tests whether the high-frequency boost actually sharpens text-to-image on the 2k overfit.

**Files:**
- Create: `tools/run_hf.sh` (sbatch or clusterrun driver)
- Uses: `train_coco64.py`, `tools/darkness_diag.py` (already weight-norm aware), the recon best `ckpt_gen_best`

**Interfaces:**
- Consumes: `--hf-weight` (Task 3), `--weight-norm`.
- Produces: brightness/contrast numbers per gamma logged to `docs/experiments/LOG.md`, and a verdict.

- [ ] **Step 1: Retrain the overfit with the HF boost, a few gammas.** Warm-start from the recon best (`ckpt_gen_best`) so the decode starts from a working reconstruction, then continue recon training with `--hf-weight` and `--weight-norm` (weight-norm keeps it stable while the HF boost amplifies high-frequency error). For each gamma in {0.5, 1.0, 2.0}: seed a ckpt dir from `ckpt_gen_best`, run `train_coco64.py --config coco64_gen --hf-weight GAMMA --weight-norm --train-mode recon --pairs 2000 --epochs 15 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --ckpt ckpt_hf_g<GAMMA> --resume` on an L4 (detached or synchronous per length). Watch that it stays finite and bounded (HF amplifies high-freq; if it destabilizes, note the gamma ceiling).

- [ ] **Step 2: Retest each with `darkness_diag`.** Run `tools/darkness_diag.py --ckpt ckpt_hf_g<GAMMA>_best --weight-norm --k 8` per gamma, and read the boosted-generation `gen/true` mean ratio (brightness) and std ratio (contrast) and the fine-scale latent ratios. The baseline to beat is the plateau brightness ~0.40 and contrast 0.24-0.40. Also fetch and eyeball the generated PNGs if `gen_retest.py` is run for the visual (recognizable, caption-specific structure).

- [ ] **Step 3: Log the verdict.** Prepend a dated entry to `docs/experiments/LOG.md` with the per-gamma brightness/contrast and whether the HF boost lifts generation off the plateau (contrast meaningfully up, images sharper) or not (points to Approach B/C next). Update `docs/STATE.md`. Commit:
```bash
git add tools/run_hf.sh docs/experiments/LOG.md docs/STATE.md
git commit -m "ran the HF-weight gamma sweep on the 2k overfit and recorded whether boosting the high-frequency decode error sharpens text-to-image or leaves it at the conditional-mean plateau"
```

---

## Notes for the implementer

- The whole change is a fixed linear high-pass on the layer's own local error, then the class's own `update_wts`. NO backprop, NO separate decoder, NO optimizer. This is the core project constraint (bidirectional PC).
- `hf_gamma` is a plain Python float so the compiled sweep's `if self.hf_gamma == 0.0` branch resolves at trace time; set it before the first training step (Task 3 does). Never make it a `tf.Variable`.
- Off (default, `hf_gamma == 0`) must be byte-identical: `_hf` returns `e`, and only the four d_pred `input=` args are wrapped. The Task 2 COCO64 GATE_MATCH is the guard; the NATIVE-143 gate is the canonical proof, deferred to a big-GPU window.
- Only the bottom conv (`prev_layer is img_input`) gets `hf_gamma` set; its d_pred error is the pixel-space error where sharpening belongs. Every other conv keeps `hf_gamma = 0` (its `_hf` is a no-op).
- This composes with `--weight-norm`; the bottom conv's `update_wts` already routes through `weight()`, so the two flags stack with no extra work. Prefer running the sweep WITH `--weight-norm` so the HF amplification does not destabilize.
