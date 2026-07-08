# Invertible (bidirectional) image downsampling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the image path's one-way max-pooling with stride-2 shared-weight convolutions so top-down (generative) drive flows through the downsampling, unblocking text→image spatial generation — config-driven so NATIVE stays byte-identical.

**Architecture:** One bidirectional PC net. A strided `Conv2DPCNLayer` downsamples via `conv2d(stride 2)` (encode/down) and upsamples via `conv2d_transpose(stride 2)` (decode/up) using the SAME weights, so it participates in the relaxation both directions (unlike the always-clamped, no-unpool maxpool). Opt-in via a `downsample` config field; a new `COCO64_GEN` config turns it on, then we retrain and re-test text→image.

**Tech Stack:** TensorFlow 2.21 (Python 3.13), the existing PCN layer classes, `tools/clusterrun.sh` (H200), pytest for local unit tests.

## Global Constraints

- The bidirectional class ONLY; ONE shared-weight image net used both directions. The strided downsampler uses the SAME `wts` forward (`conv2d`) and top-down (`conv2d_transpose`). NO separate decoder, NO backprop through the net. Weight learning stays the existing local beta-less LARS (unchanged).
- Config-driven / opt-in: conv `stride` defaults to **1** and `downsample` defaults to **`'maxpool'`**, so `NATIVE_7B` (and existing `COCO64_156M`) are byte-identical and `NATIVE_7B` still reproduces `GATE_MATCH nlayers=143`. NEVER relax the gate.
- The five shared-latent aliases stay. All existing conv output sizes, flatten→dense projections, and the 5 multi-scale taps (off conv2/4/6/8/9) stay identical in shape. Downsamplers are channel-preserving (in=out=preceding conv's channels), kernel 2×2, SAME padding, `'linear'`.
- Commits are first-person student voice, NO AI attribution / Co-Authored-By / "Generated with". Commit locally; the controller pushes at checkpoints.
- Cluster gates use `tools/clusterrun.sh` (Bash tool timeout 600000 ms). `clusterrun` CANNOT take an inline `python3 -c` with single quotes — put throwaway scripts in files and sync them.

---

## File Structure

- `conv_pcn_layer.py` (Task 1) — add `stride` support to `Conv2DPCNLayer` (default 1, inert).
- `pcn_config.py` (Task 2) — add `downsample` field + `COCO64_GEN` config.
- `encoder_encoder_pcn.py` (Task 3) — constructor branch: build maxpool or strided downsampler at each of the 4 sites.
- `train_coco64.py` (Task 4) — `--config` selection; keep downsamplers linear under the conv-activation override.
- (Task 5) — retrain `COCO64_GEN` + text→image retest (controller-driven run; no code commit).

---

## Task 1: `stride` support in `Conv2DPCNLayer`

**Files:**
- Modify: `conv_pcn_layer.py`
- Test: `tests/test_conv_stride.py`

**Interfaces:**
- Produces: `Conv2DPCNLayer(..., stride:int=1)`. With `stride=2`, forward halves H,W (SAME) and `predict_prev()` restores the pre-downsample H,W,C. `stride=1` is byte-identical to today.

- [ ] **Step 1: Write the failing test** `tests/test_conv_stride.py`:

```python
import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer

def _forward(L, x):
    return L(tf.convert_to_tensor(x), set_state=True)

def test_stride2_halves_and_transpose_restores():
    x = tf.random.normal((2, 8, 8, 3))
    L = Conv2DPCNLayer(5, (2, 2), 1e-4, 'linear', padding='SAME', stride=2)
    out = _forward(L, x)
    assert tuple(out.shape) == (2, 4, 4, 5)          # stride-2 SAME halves H,W
    pp = L.predict_prev()
    assert tuple(pp.shape) == (2, 8, 8, 3)           # transpose restores input H,W,C

def test_stride1_default_same_padding_preserves():
    x = tf.random.normal((2, 8, 8, 3))
    L = Conv2DPCNLayer(5, (3, 3), 1e-4, 'linear', padding='SAME')   # default stride=1
    out = _forward(L, x)
    assert tuple(out.shape) == (2, 8, 8, 5)
    pp = L.predict_prev()
    assert tuple(pp.shape) == (2, 8, 8, 3)
    assert L.stride == 1
```

- [ ] **Step 2: Run, verify fail** — `cd ~/encoder_encoder_PCN && python3 -m pytest tests/test_conv_stride.py -q` → FAIL (`stride` not accepted / attr missing).

- [ ] **Step 3: Thread `stride` through `Conv2DPCNLayer`.**

In `__init__` (line 17), add a `stride` parameter and attribute (default 1):
```python
    def __init__(self, num_units:int, kernel_size:tuple[int, int], learning_rate:float, activation:Literal['linear', 'relu']='linear', prev_layer:object=None, next_layers:list=None, padding:str='VALID', stride:int=1):
```
and after `self.padding = padding` add:
```python
        self.stride = stride
        self.input_shape = None   # pre-downsample spatial shape, recorded at forward time (for the strided transpose)
```

In `net_in` (line 178), record the input shape and use the stride:
```python
    def net_in(self, x:tf.Tensor):
        if self.wts is None:
            self.init_params(x.shape)
        self.input_shape = x.shape
        return tf.nn.conv2d(x, self.wts, padding=self.padding, strides=self.stride)
```

In `predict_prev` (line 41), keep the `stride == 1` path EXACTLY as today (byte-identical → gate-safe) and add a strided branch that restores the recorded input size:
```python
    def predict_prev(self):
        if self.stride == 1:
            if self.padding == 'SAME':
                output_shape = (self.output_shape[0], self.output_shape[1], self.output_shape[2], self.wts.shape[-2])
            else:
                output_shape = (self.output_shape[0], self.output_shape[1]+self.kernel_size[0]-1, self.output_shape[2]+self.kernel_size[1]-1, self.wts.shape[-2])
            return tf.nn.conv2d_transpose(self.state, self.wts, padding=self.padding, strides=1, output_shape=output_shape)
        else:
            output_shape = (self.output_shape[0], self.input_shape[1], self.input_shape[2], self.wts.shape[-2])
            return tf.nn.conv2d_transpose(self.state, self.wts, padding=self.padding, strides=self.stride, output_shape=output_shape)
```

In `pred_loss_d_input` (lines 56-64), change each `strides=1` to `strides=self.stride` (output_shape stays `x.shape`, already correct). All four branches.

In `update_state`, the bottom-up `tf.nn.conv2d(...)` calls (the `d_pred += tf.nn.conv2d(..., strides=1, padding=self.padding)` lines in the relu/gelu/silu/else branches, ~lines 107-120): change `strides=1` → `strides=self.stride`.

In `update_wts` (lines 132-166), change every `strides=[1, 1, 1, 1]` (the `Conv2DBackpropFilter` calls, both the `d_state` and `d_pred` groups) to `strides=[1, self.stride, self.stride, 1]`.

Do NOT change any other logic. With `stride=1` (the default) every call is literally unchanged.

- [ ] **Step 4: Run, verify pass** — `python3 -m pytest tests/test_conv_stride.py -q` → 2 pass. (TF-CPU local is fine.)

- [ ] **Step 5: NATIVE gate (cluster, H200, Bash timeout 600000 ms).** The conv change is inert at `stride=1`:
```
tools/clusterrun.sh --name stride_gate --gpu H200 --mem 96G --cpus 4 --time 00:30:00 --sync "conv_pcn_layer.py encoder_encoder_pcn.py pcn_config.py dense_pcn_layer.py transformer_pcn_layer.py tools/rewrite_gate.py tools/gate_compare.py" --run "python3 tools/rewrite_gate.py --steps 2 --save golden_stride.npz && python3 tools/gate_compare.py golden_baseline.npz golden_stride.npz"
```
Expect `GATE_MATCH nlayers=143`. If mismatch, a `stride=1` path was altered — restore it byte-identical.

- [ ] **Step 6: Commit**
```bash
git add conv_pcn_layer.py tests/test_conv_stride.py
git commit -m "added stride support to the conv layer so it can downsample bidirectionally (conv2d down, transpose-conv up, same weights); default stride 1 leaves NATIVE byte-identical"
```

---

## Task 2: `downsample` config + `COCO64_GEN`

**Files:**
- Modify: `pcn_config.py`
- Test: `tests/test_pcn_config.py` (extend)

**Interfaces:**
- Produces: `PCNConfig.downsample` (default `'maxpool'`, validated to `{'maxpool','strided_conv'}`); `COCO64_GEN` = `COCO64_156M` with `downsample='strided_conv'`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_pcn_config.py`:
```python
def test_downsample_field_and_coco64_gen():
    from pcn_config import PCNConfig, NATIVE_7B, COCO64_156M, COCO64_GEN
    assert NATIVE_7B.downsample == 'maxpool'
    assert COCO64_156M.downsample == 'maxpool'
    assert COCO64_GEN.downsample == 'strided_conv'
    # COCO64_GEN matches COCO64_156M in every other field
    for f in vars(COCO64_156M):
        if f != 'downsample':
            assert getattr(COCO64_GEN, f) == getattr(COCO64_156M, f)

def test_downsample_validation():
    import pytest
    from pcn_config import PCNConfig, COCO64_156M
    import dataclasses
    with pytest.raises(Exception):
        dataclasses.replace(COCO64_156M, downsample='bogus')
```

- [ ] **Step 2: Run, verify fail** — `python3 -m pytest tests/test_pcn_config.py -q` → FAIL (no `downsample` / no `COCO64_GEN`).

- [ ] **Step 3: Add the field + validation + config.** In `pcn_config.py` add to the `PCNConfig` dataclass a field `downsample: str = 'maxpool'` (place it with a default so existing positional construction is unaffected). In `__post_init__`, add:
```python
        assert self.downsample in ('maxpool', 'strided_conv'), f"downsample must be maxpool|strided_conv, got {self.downsample}"
```
After the `COCO64_156M` definition add:
```python
import dataclasses as _dc
COCO64_GEN = _dc.replace(COCO64_156M, downsample='strided_conv')
```
(If `COCO64_156M` is built via `PCNConfig(...)`, `dataclasses.replace` copies all fields and overrides `downsample`; confirm `COCO64_GEN` is importable.)

- [ ] **Step 4: Run, verify pass** — `python3 -m pytest tests/test_pcn_config.py -q` → all pass.

- [ ] **Step 5: Commit**
```bash
git add pcn_config.py tests/test_pcn_config.py
git commit -m "added a downsample config field (maxpool default) and a COCO64_GEN config that turns on strided-conv downsampling; NATIVE and COCO64_156M keep maxpool"
```

---

## Task 3: constructor branch — build the downsampler per config

**Files:**
- Modify: `encoder_encoder_pcn.py`
- Test: (cluster) NATIVE GATE_MATCH + COCO64_GEN structural check

**Interfaces:**
- Consumes: `Conv2DPCNLayer(..., stride=2)` (Task 1), `config.downsample` (Task 2).
- Produces: with `downsample='strided_conv'`, the 4 maxpools are replaced by stride-2 channel-preserving convs appended to `trainable_layers`; wiring (prev/next + taps) unchanged.

- [ ] **Step 1: Add a downsample helper and use it at the 4 sites.** In `__init__`, before `conv1` is built, define a local helper:
```python
        def _build_downsample(prev_conv, channels):
            if config.downsample == 'strided_conv':
                ds = Conv2DPCNLayer(channels, (2, 2), learning_rate, 'linear', prev_conv, padding='SAME', stride=2)
                self.trainable_layers.append(ds)
                return ds
            return MaxPool2DPCNLayer((2, 2), prev_conv)
```
Then replace each of the four `mpN = MaxPool2DPCNLayer((2, 2), convK)` lines with `mpN = _build_downsample(convK, config.conv_channels[K-1])`, keeping the surrounding `convK.next_layers = [mpN]` / `mpN.next_layers = [...]` / tap-append lines EXACTLY as they are:
- `mp1 = _build_downsample(conv2, config.conv_channels[1])`
- `mp2 = _build_downsample(conv4, config.conv_channels[3])`
- `mp3 = _build_downsample(conv6, config.conv_channels[5])`
- `mp4 = _build_downsample(conv8, config.conv_channels[7])`

Channels are the preceding conv's channel count (channel-preserving, like the pool). Do NOT change any other wiring, the flatten taps, or the dense/text path.

- [ ] **Step 2: NATIVE gate + COCO64_GEN structural check (cluster, H200, Bash timeout 600000 ms).** Create throwaway `tools/_ds_check.py`:
```python
import tensorflow as tf
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import COCO64_GEN as G
def finite(m):
    return [i for i,L in enumerate(m.trainable_layers)
            if getattr(L,'state',None) is not None and not bool(tf.reduce_all(tf.math.is_finite(tf.cast(L.state,tf.float32))))]
m = EncoderEncoderPCN(1e-4, config=G)
B = 4
img = tf.random.normal((B, G.img_resolution, G.img_resolution, 3), seed=0)
txt = tf.random.normal((B, G.txt_seq_len, G.txt_embed_dim), seed=0)
mask = tf.zeros((B, G.txt_seq_len))
m.img_input.is_clamped = True; m.txt_input.is_clamped = True
m.pass_through(img, txt, mask)
m.update_states_wts_b_relaxed(2, 5)
st = [id(getattr(L,'state',None)) for L in m.trainable_layers if getattr(L,'state',None) is not None]
print("COCO64_GEN nonfinite=", finite(m), "aliases=", len(st)-len(set(st)), "(expect 5)", flush=True)
oi = m.test_step(10, img[:1], txt[:1], predict='img', mask=mask[:1])
ot = m.test_step(10, img[:1], txt[:1], predict='txt', mask=mask[:1])
print("GEN_IMG finite=", bool(tf.reduce_all(tf.math.is_finite(oi))), "GEN_TXT finite=", bool(tf.reduce_all(tf.math.is_finite(ot))), flush=True)
nparams = int(sum(int(tf.size(l.wts)) for l in m.trainable_layers if getattr(l,'wts',None) is not None))
print("COCO64_GEN conv/dense weight params=", nparams, flush=True)
print("DS_CHECK_DONE", flush=True)
```
Run:
```
tools/clusterrun.sh --name ds_check --gpu H200 --mem 96G --cpus 4 --time 00:30:00 --sync "conv_pcn_layer.py encoder_encoder_pcn.py pcn_config.py dense_pcn_layer.py transformer_pcn_layer.py coco64_data.py tools/_ds_check.py tools/rewrite_gate.py tools/gate_compare.py" --run "python3 tools/rewrite_gate.py --steps 2 --save golden_ds.npz && python3 tools/gate_compare.py golden_baseline.npz golden_ds.npz && python3 tools/_ds_check.py"
```
Expect: `GATE_MATCH nlayers=143` (NATIVE maxpool path unchanged), `COCO64_GEN nonfinite= []`, `aliases= 5`, both `GEN_IMG/GEN_TXT finite= True`. Delete `tools/_ds_check.py` after (do not commit). If the gate mismatches, the maxpool branch was altered — restore it.

- [ ] **Step 3: Commit**
```bash
git add encoder_encoder_pcn.py
git commit -m "build a stride-2 bidirectional conv in place of each image maxpool when downsample=strided_conv, so top-down generative drive flows through; maxpool path unchanged so NATIVE still GATE_MATCHes"
```

---

## Task 4: config selection in `train_coco64.py`

**Files:**
- Modify: `train_coco64.py`
- Test: (cluster) COCO64_GEN training smoke

**Interfaces:**
- Produces: `--config {coco64_156m,coco64_gen}` (default `coco64_156m`). The conv-activation override applies to stride-1 convs only, so strided downsamplers keep their `'linear'` activation.

- [ ] **Step 1: Add `--config` + the config map + import.** Add `from pcn_config import COCO64_156M, COCO64_GEN` (keep the existing import), then:
```python
    ap.add_argument("--config", default="coco64_156m", choices=["coco64_156m", "coco64_gen"])
```
Replace `m = EncoderEncoderPCN(a.lr, config=COCO64_156M)` with:
```python
    CONFIGS = {"coco64_156m": COCO64_156M, "coco64_gen": COCO64_GEN}
    m = EncoderEncoderPCN(a.lr, config=CONFIGS[a.config])
    print(f"config={a.config}", flush=True)
```

- [ ] **Step 2: Keep downsamplers linear under the conv-activation override.** In the `if a.conv_activation != "relu":` block, change the loop condition so only stride-1 convs are overridden:
```python
        for L in m.trainable_layers:
            if isinstance(L, Conv2DPCNLayer) and getattr(L, "stride", 1) == 1:
                L.activation = a.conv_activation; nc += 1
```
(The stride-2 downsamplers stay `'linear'`, as designed.)

- [ ] **Step 3: COCO64_GEN training smoke (cluster, H200, Bash timeout 600000 ms).**
```
tools/clusterrun.sh --name gen_smoke --gpu H200 --mem 96G --cpus 4 --time 00:25:00 --sync "train_coco64.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py coco64_data.py" --run "python3 train_coco64.py --config coco64_gen --pairs 64 --epochs 4 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --energy-every 5 --ckpt ckpt_gen_smoke"`
```
Expect: `config=coco64_gen`, energy lines finite + descending-ish, `max_abs_state` under 400, no `DIVERGED`/NaN, `TRAIN_DONE`. (Downsamplers train; this only checks stability/finiteness, not generation quality.) Leave `ckpt_gen_smoke*` unstaged.

- [ ] **Step 4: Commit**
```bash
git add train_coco64.py
git commit -m "added a --config selector to train on COCO64_GEN (strided-conv downsampling), and kept the strided downsamplers linear under the conv-activation override"
```

---

## Task 5: retrain `COCO64_GEN` + text→image retest (deliverable)

**Files:** none (controller-driven run of Task 4's script + a throwaway generation retest).

- [ ] **Step 1: Launch the 2k retrain** (detached sbatch on the H200; the stable recipe). Sync the files, then submit a job script running:
`python3 train_coco64.py --config coco64_gen --pairs 2000 --epochs 15 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --energy-every 50 --ckpt ckpt_gen` (env exports = the PATH/LD_LIBRARY_PATH/PYTHONPATH block). Monitor energy + max|state|; if it diverges, note it (a real datum) and fall back to a lower LR — but the recipe is the known-stable one.

- [ ] **Step 2: text→image retest on `ckpt_gen_best`.** Write a throwaway (like the earlier retest, config `COCO64_GEN`, gelu, state_clip 400, 150 relax): for a few in-sample pairs, generate text→image (from a zero image init) and image→caption via the model's OWN `test_step` (no manual top-down boost — the pathway is now reconnected), save PNGs, print generated-image participation ratio + PNG byte sizes. Run via `clusterrun --fetch`. THE KEY check: does text→image now show caption-VARYING STRUCTURE (image PR well above 1; PNGs differ by caption and show scene content), with reconstruction + image→caption still intact? Read a few fetched PNGs to judge visually. Delete the throwaway after.

- [ ] **Step 3: Record the outcome** in an SP report: final energy/stability, whether text→image moved to structured caption-varying images (PR + visual read), image→caption + reconstruction status. No code commit; the deliverable is the checkpoint + the verdict on whether invertible downsampling unblocks text→image. If it does NOT (still coarse/uniform even with the pathway reconnected), that localizes the remaining obstacle to the shared latent / capacity, which is the next question.

---

## Plan exit criteria

Invertible strided-conv downsampling shipped opt-in (NATIVE unchanged, `GATE_MATCH nlayers=143`; `COCO64_GEN` builds and both generation directions run), trained on COCO64, with a clear read on whether text→image now produces caption-varying structure. Held-out, multi-scale InfoNCE, and anti-checkerboard kernel tuning are follow-ups.

## Self-Review

- **Spec coverage:** Component 1 (strided conv) = Task 1; Component 2 (config) = Task 2; Component 3 (constructor branch) = Task 3; Component 4 (retrain + eval) = Tasks 4–5; validation/gates = Task 1 Step 5 (NATIVE gate), Task 3 Step 2 (NATIVE gate + COCO64_GEN structural), Task 1 Step 1 (shape round-trip). No gaps.
- **Placeholder scan:** all steps carry exact code/commands; Task 5 is an explicit run + throwaway retest (described, deleted, not committed), not a placeholder.
- **Type consistency:** `Conv2DPCNLayer(..., stride=1)` defined in Task 1, used with `stride=2` in Task 3; `PCNConfig.downsample` / `COCO64_GEN` defined in Task 2, read in Task 3 (`config.downsample`) and Task 4 (`--config`); `getattr(L,'stride',1)` used in Task 4 matches the attr added in Task 1.
