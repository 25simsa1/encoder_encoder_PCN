# Phase 3: 64px Config-Driven Retarget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `EncoderEncoderPCN` build from a `PCNConfig` object, ship a `NATIVE_7B` config that reproduces today's 572px/7.7B model exactly, and add a `COCO64_156M` config that lands near 156M params and generates in both directions at 64px.

**Architecture:** Extract every size literal in the ~330-line `__init__` into a config object, refactoring image path and text path separately so each half is proven non-breaking by the existing 572px golden gate. Then add and tune the 64px config. The conv depth, the five multi-scale taps, and the five shared-latent state pairs are kept structurally; only widths and the input resolution move into config.

**Tech Stack:** TensorFlow 2.21, the Colby HPC cluster (H200 via `tools/clusterrun.sh`), the existing `~/tf-env`, pytest for the local config test.

## Global Constraints

- The model is the bidirectional class only (`encoder_encoder_pcn.py` + `conv/dense/transformer_pcn_layer.py`). PC learning only, no functional-version code, no backprop, no pretrained parts.
- The five shared-latent pairs must stay aliased state Variables: image `dense2/6/10/14/18` share state with text `dense4/8/12/16/20`, so within a scale the image and text shared-latent dims MUST be equal.
- `NATIVE_7B` must reproduce the current model byte-for-byte, proven by `GATE_MATCH` against the existing `golden_baseline.npz` (rel tol 1e-4). This is the safety gate for the whole refactor. Construction ORDER and the `trainable_layers` order must be preserved (the gate signature is keyed by layer index; kaiming init depends on construction order + the seed).
- `__init__` stays backward compatible: `EncoderEncoderPCN(learning_rate, config=NATIVE_7B)` with `NATIVE_7B` the default, so `EncoderEncoderPCN(1e-4)` and every existing caller (the gate, `train_step`, the compiled sweeps + their guards) keep working unchanged.
- All GPU runs use `tools/clusterrun.sh` on the H200. Commits: first-person student voice, NO AI attribution / Co-Authored-By / "Generated with". Commit locally; the controller pushes at checkpoints.

## File Structure

- Create `pcn_config.py`: the `PCNConfig` dataclass and the `NATIVE_7B` / `COCO64_156M` config instances. One responsibility, importable by the model and by tools.
- Modify `encoder_encoder_pcn.py`: `__init__` gains a `config` param and builds all widths/shapes from it (image path in Task 2, text path in Task 3). No behavior change for `NATIVE_7B`.
- Modify `tools/rewrite_gate.py`: add a `--config {native7b,coco64}` flag so the gate can build either config (used from Task 4 on).
- Create `tools/count_params.py`: instantiate a config, run `pass_through` (to realize the lazy weight Variables), and print total + per-layer params. Used to tune `COCO64_156M`.
- Create `tests/test_pcn_config.py`: local unit test for the config objects (no model build, no GPU).

---

### Task 1: `PCNConfig` dataclass + `NATIVE_7B` and `COCO64_156M` configs

**Files:**
- Create: `pcn_config.py`
- Test: `tests/test_pcn_config.py`

**Interfaces:**
- Produces: `PCNConfig` (frozen dataclass); `NATIVE_7B: PCNConfig`; `COCO64_156M: PCNConfig`. Later tasks read `config.img_resolution`, `config.conv_channels`, `config.inter_dim`, `config.img_dense_relu_widths` (5-tuple, tap order conv9,conv8,conv6,conv4,conv2), `config.shared_latent_dims` (5-tuple, same order, shared image<->text), `config.txt_seq_len`, `config.txt_embed_dim`, `config.txt_group_widths` (4-tuple), `config.txt_group_blocks` (4-tuple), `config.txt_heads`, `config.txt_sublayers`, `config.txt_bridge_seq_lens` (3-tuple), `config.txt_dense_relu_widths` (5-tuple, tap order).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pcn_config.py
from pcn_config import PCNConfig, NATIVE_7B, COCO64_156M

def test_native_reproduces_current_literals():
    c = NATIVE_7B
    assert c.img_resolution == 572
    assert c.conv_channels == (64, 64, 128, 128, 256, 256, 512, 512, 1024)
    assert c.inter_dim == 100
    # tap order conv9, conv8, conv6, conv4, conv2
    assert c.img_dense_relu_widths == (307200, 582542, 1279723, 2654815, 5433667)
    assert c.shared_latent_dims == (102400, 161817, 345871, 702332, 1429912)
    assert c.txt_seq_len == 192 and c.txt_embed_dim == 512
    assert c.txt_group_widths == (512, 1024, 2048, 4096)
    assert c.txt_group_blocks == (3, 3, 3, 8)
    assert c.txt_heads == 8 and c.txt_sublayers == 3
    assert c.txt_bridge_seq_lens == (48, 12, 3)
    assert c.txt_dense_relu_widths == (36864, 44237, 90931, 185795, 373555)

def test_shared_dims_are_the_shared_contract():
    # the five shared-latent dims are what image dense2/6/10/14/18 and text
    # dense4/8/12/16/20 both use; a single tuple guarantees they match.
    assert len(NATIVE_7B.shared_latent_dims) == 5
    assert len(COCO64_156M.shared_latent_dims) == 5

def test_coco64_is_64px_and_smaller():
    c = COCO64_156M
    assert c.img_resolution == 64
    assert c.conv_channels == NATIVE_7B.conv_channels  # depth/channels unchanged
    assert c.inter_dim == 100
    # initial estimate (Task 4 tunes these); must be far smaller than native
    assert max(c.shared_latent_dims) <= max(NATIVE_7B.shared_latent_dims) // 10
    assert c.txt_seq_len == 32
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cd ~/encoder_encoder_PCN && ~/tf-env/bin/python3 -m pytest tests/test_pcn_config.py -q` (or system `python3 -m pytest` if tf-env not local)
Expected: FAIL, `ModuleNotFoundError: No module named 'pcn_config'`.

- [ ] **Step 3: Write `pcn_config.py`**

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PCNConfig:
    name: str
    img_resolution: int
    conv_channels: tuple      # conv1..conv9 output channels
    inter_dim: int            # bottleneck width (the inter(100) layers)
    img_dense_relu_widths: tuple   # 5, tap order conv9,conv8,conv6,conv4,conv2
    shared_latent_dims: tuple      # 5, same order; shared image<->text
    txt_seq_len: int
    txt_embed_dim: int
    txt_group_widths: tuple   # transformer width per group
    txt_group_blocks: tuple   # transformer blocks per group
    txt_heads: int
    txt_sublayers: int        # the first arg to TransformerPCNLayer
    txt_bridge_seq_lens: tuple     # linear_2/4/6 sequence reductions between groups
    txt_dense_relu_widths: tuple   # 5, text tap dense_relu (dense3/7/11/15/19)

NATIVE_7B = PCNConfig(
    name="native7b",
    img_resolution=572,
    conv_channels=(64, 64, 128, 128, 256, 256, 512, 512, 1024),
    inter_dim=100,
    img_dense_relu_widths=(307200, 582542, 1279723, 2654815, 5433667),
    shared_latent_dims=(102400, 161817, 345871, 702332, 1429912),
    txt_seq_len=192, txt_embed_dim=512,
    txt_group_widths=(512, 1024, 2048, 4096),
    txt_group_blocks=(3, 3, 3, 8),
    txt_heads=8, txt_sublayers=3,
    txt_bridge_seq_lens=(48, 12, 3),
    txt_dense_relu_widths=(36864, 44237, 90931, 185795, 373555),
)

# Initial estimate (~160M); Task 4 tunes to ~156M. Shared dims are an ~8x
# compression of each 64px tap feature map, keeping the native ordering
# (bigger latent for the shallow high-res taps).
COCO64_156M = PCNConfig(
    name="coco64",
    img_resolution=64,
    conv_channels=(64, 64, 128, 128, 256, 256, 512, 512, 1024),
    inter_dim=100,
    img_dense_relu_widths=(2048, 4096, 8192, 16384, 32768),
    shared_latent_dims=(2048, 4096, 8192, 16384, 32768),
    txt_seq_len=32, txt_embed_dim=512,
    txt_group_widths=(512, 512, 512, 512),
    txt_group_blocks=(1, 1, 1, 3),
    txt_heads=8, txt_sublayers=3,
    txt_bridge_seq_lens=(16, 8, 4),
    txt_dense_relu_widths=(2048, 4096, 8192, 8192, 8192),
)
```

- [ ] **Step 4: Run test, verify pass**

Run: `~/tf-env/bin/python3 -m pytest tests/test_pcn_config.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add pcn_config.py tests/test_pcn_config.py
git commit -m "added a PCNConfig with a NATIVE_7B config holding the current literal widths and an initial COCO64_156M config for the 64px retarget"
```

---

### Task 2: Build the IMAGE path from config; NATIVE_7B still GATE_MATCHes

**Files:**
- Modify: `encoder_encoder_pcn.py` (`__init__` signature + the image-path build, roughly lines 130-270)

**Interfaces:**
- Consumes: `PCNConfig`, `NATIVE_7B` from Task 1.
- Produces: `EncoderEncoderPCN(learning_rate, config=NATIVE_7B)` — new optional `config` param, default `NATIVE_7B`.

- [ ] **Step 1: Add the config param and refactor the image path**

Change the constructor to `def __init__(self, learning_rate, config=NATIVE_7B):` (import from `pcn_config`), store `self.config = config`. Then replace every hardcoded IMAGE-path literal with the matching config field, preserving construction order and wiring exactly:
- conv channels `64,64,128,...,1024` come from `config.conv_channels[0..8]`.
- the input shape uses `config.img_resolution` (see how `run_instrumented.py` / `pass_through` feed `(1,572,572,3)`; the resolution only affects the fed tensors + the flatten sizes, which are derived from the conv output shapes automatically — do NOT hardcode flatten sizes).
- the `inter(100)` layers use `config.inter_dim`.
- the five image `dense_relu` widths (dense1,dense5,dense9,dense13,dense17) come from `config.img_dense_relu_widths` in tap order conv9,conv8,conv6,conv4,conv2.
- the five image shared layers (dense2,dense6,dense10,dense14,dense18) use `config.shared_latent_dims` in the same order.
Do NOT reorder layers, rename them, change `next_layers` wiring, or touch the text path yet. This is a mechanical literal->field substitution.

- [ ] **Step 2: Validate NATIVE_7B on the cluster (GATE_MATCH)**

Run (Bash tool timeout 600000 ms):
```
tools/clusterrun.sh --name p3_img_gate --gpu H200 --mem 96G --cpus 4 --time 00:30:00 \
  --sync "encoder_encoder_pcn.py pcn_config.py tools/rewrite_gate.py tools/gate_compare.py" \
  --run "python3 tools/rewrite_gate.py --steps 2 --save golden_native_img.npz && python3 tools/gate_compare.py golden_baseline.npz golden_native_img.npz"
```
Expected: log ends with `GATE_MATCH nlayers=143`. This proves the image-path refactor changed nothing about the model (the gate builds the default `NATIVE_7B`). If `GATE_MISMATCH`, the refactor altered a width/order/wiring — fix it; do NOT relax the gate.

- [ ] **Step 3: Commit**

```bash
git add encoder_encoder_pcn.py
git commit -m "built the image path from PCNConfig (widths, channels, inter dim, resolution). NATIVE_7B still GATE_MATCHes the 572px golden, so the refactor changed nothing"
```

---

### Task 3: Build the TEXT path from config; NATIVE_7B still GATE_MATCHes

**Files:**
- Modify: `encoder_encoder_pcn.py` (the text-path build, roughly lines 267-460)

**Interfaces:**
- Consumes: `PCNConfig` text fields from Task 1.
- Produces: a fully config-driven constructor (both paths).

- [ ] **Step 1: Refactor the text path to read config**

Replace the text-path literals with config fields, preserving the pyramid structure and wiring exactly:
- `txt_input` feeds an embedding sized `config.txt_embed_dim`; the fed text tensor is `(batch, config.txt_seq_len, config.txt_embed_dim)`.
- the four transformer groups use `config.txt_group_widths[g]` as width and `config.txt_group_blocks[g]` as the block count, with `config.txt_heads` heads and `config.txt_sublayers` (the first `TransformerPCNLayer` arg). Build groups with a loop rather than 17 hardcoded blocks.
- the between-group linear/transpose bridges use `config.txt_group_widths` for the width-up linears and `config.txt_bridge_seq_lens` for the sequence-reduction linears (the 48,12,3 values).
- the five text `dense_relu` widths (dense3,dense7,dense11,dense15,dense19) come from `config.txt_dense_relu_widths`.
- the five text shared layers (dense4,dense8,dense12,dense16,dense20) use `config.shared_latent_dims` (SAME tuple as image) and keep `share_state_layer=dense2/6/10/14/18` exactly.
If capturing the pyramid faithfully needs one or two more config fields, add them to `PCNConfig` and to `NATIVE_7B` with the current literal values (and extend `tests/test_pcn_config.py`); GATE_MATCH is the correctness gate.

- [ ] **Step 2: Validate NATIVE_7B on the cluster (GATE_MATCH)**

Run (Bash tool timeout 600000 ms):
```
tools/clusterrun.sh --name p3_txt_gate --gpu H200 --mem 96G --cpus 4 --time 00:30:00 \
  --sync "encoder_encoder_pcn.py pcn_config.py tools/rewrite_gate.py tools/gate_compare.py" \
  --run "python3 tools/rewrite_gate.py --steps 2 --save golden_native_txt.npz && python3 tools/gate_compare.py golden_baseline.npz golden_native_txt.npz"
```
Expected: `GATE_MATCH nlayers=143`. The whole constructor is now config-driven and provably reproduces the 7.7B model. If mismatch, the text refactor deviated — fix it.

- [ ] **Step 3: Commit**

```bash
git add encoder_encoder_pcn.py pcn_config.py tests/test_pcn_config.py
git commit -m "built the text path from PCNConfig too (transformer groups as a loop, bridge widths/seq-lens, shared dims from the same tuple as the image side). NATIVE_7B still GATE_MATCHes: the whole constructor is now config-driven with zero change to the model"
```

---

### Task 4: `count_params.py` + `--config` gate flag + tune COCO64_156M to ~156M

**Files:**
- Create: `tools/count_params.py`
- Modify: `tools/rewrite_gate.py` (add `--config`)
- Modify: `pcn_config.py` (tune `COCO64_156M` widths only)

**Interfaces:**
- Consumes: `EncoderEncoderPCN(config=...)`, `NATIVE_7B`, `COCO64_156M`.
- Produces: `tools/count_params.py` printing `TOTAL_PARAMS=<n>`; `rewrite_gate.py --config {native7b,coco64}`.

- [ ] **Step 1: Write `tools/count_params.py`**

```python
import argparse
import tensorflow as tf, numpy as np
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import NATIVE_7B, COCO64_156M

CFG = {"native7b": NATIVE_7B, "coco64": COCO64_156M}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="coco64")
    a = ap.parse_args(); cfg = CFG[a.config]
    r = cfg.img_resolution
    m = EncoderEncoderPCN(1e-4, config=cfg)
    img = tf.zeros((1, r, r, 3)); txt = tf.zeros((1, cfg.txt_seq_len, cfg.txt_embed_dim)); mask = tf.zeros((1, cfg.txt_seq_len))
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)   # realize lazy weight Variables
    P = 0
    for L in m.trainable_layers:
        for k, v in vars(L).items():
            if isinstance(v, tf.Variable) and k != "state":
                P += int(np.prod(v.shape))
    print(f"TOTAL_PARAMS={P} ({P/1e6:.1f}M) config={cfg.name} nlayers={len(m.trainable_layers)}", flush=True)
```

- [ ] **Step 2: Add `--config` to `tools/rewrite_gate.py`**

In `run_reference`, accept `config_name="native7b"`, map to the config object, and pass `config=` to `EncoderEncoderPCN(...)` and use `cfg.img_resolution`/`txt_seq_len`/`txt_embed_dim` for the input tensor shapes. Add argparse `--config` (default `native7b` so existing behavior is unchanged). Keep the default path identical to today.

- [ ] **Step 3: Count COCO64 params on the cluster**

Run (Bash tool timeout 600000 ms; a 64px model is small so this is fast):
```
tools/clusterrun.sh --name p3_count --gpu H200 --mem 32G --cpus 4 --time 00:15:00 \
  --sync "tools/count_params.py pcn_config.py encoder_encoder_pcn.py" \
  --run "python3 tools/count_params.py --config coco64"
```
Expected: a `TOTAL_PARAMS=... (XXX.XM)` line.

- [ ] **Step 4: Tune `COCO64_156M` widths to land in [125M, 190M]**

If the count is outside [125M, 190M], edit ONLY the `COCO64_156M` width fields in `pcn_config.py` (cheapest levers first: `txt_group_blocks`, `txt_group_widths`, then `img_dense_relu_widths`/`shared_latent_dims`) and re-run Step 3 until in range. Keep `shared_latent_dims` matched image<->text (it is a single tuple, so this is automatic). Record the final count.

- [ ] **Step 5: Commit**

```bash
git add tools/count_params.py tools/rewrite_gate.py pcn_config.py
git commit -m "added a param counter + a --config flag to the gate, and tuned COCO64_156M to land near 156M (final count in the commit body)"
```

---

### Task 5: COCO64 GPU validation — finite batched step, shared-latent aliasing, both generation directions

**Files:**
- Create: `tools/validate_coco64.py`

**Interfaces:**
- Consumes: `EncoderEncoderPCN(config=COCO64_156M)`.

- [ ] **Step 1: Write `tools/validate_coco64.py`**

```python
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import COCO64_156M as C

def finite(m):
    bad = [i for i, L in enumerate(m.trainable_layers)
           if getattr(L, "state", None) is not None
           and (bool(tf.reduce_any(tf.math.is_nan(L.state))) or bool(tf.reduce_any(tf.math.is_inf(L.state))))]
    return bad

# 1) batched relaxed step, states finite
m = EncoderEncoderPCN(1e-4, config=C)
B = 4
img = tf.random.normal((B, C.img_resolution, C.img_resolution, 3), seed=0)
txt = tf.random.normal((B, C.txt_seq_len, C.txt_embed_dim), seed=0)
mask = tf.zeros((B, C.txt_seq_len))
m.img_input.is_clamped = True; m.txt_input.is_clamped = True
m.pass_through(img, txt, mask)
m.update_states_wts_b_relaxed(2, 5)
print(f"BATCHED_RELAXED_STEP nonfinite_layers={finite(m)}", flush=True)

# 2) shared-latent aliasing: image dense2/6/10/14/18 share state with text dense4/8/12/16/20.
# They are the layers whose .state IS another layer's .state; assert 5 aliased pairs exist.
states = [id(getattr(L, "state", None)) for L in m.trainable_layers if getattr(L, "state", None) is not None]
n_shared = len(states) - len(set(states))
print(f"SHARED_STATE_ALIASES={n_shared} (expect 5)", flush=True)

# 3) both generation directions on a fresh model
mg = EncoderEncoderPCN(1e-4, config=C)
oi = mg.test_step(10, img[:1], txt[:1], predict='img', mask=mask[:1])
ot = mg.test_step(10, img[:1], txt[:1], predict='txt', mask=mask[:1])
fi = bool(tf.reduce_all(tf.math.is_finite(oi))); ft = bool(tf.reduce_all(tf.math.is_finite(ot)))
print(f"GEN_IMG shape={tuple(oi.shape)} finite={fi}", flush=True)
print(f"GEN_TXT shape={tuple(ot.shape)} finite={ft}", flush=True)
print("VALIDATE_COCO64_DONE", flush=True)
```

- [ ] **Step 2: Run on the cluster**

Run (Bash tool timeout 600000 ms):
```
tools/clusterrun.sh --name p3_validate --gpu H200 --mem 64G --cpus 4 --time 00:25:00 \
  --sync "tools/validate_coco64.py pcn_config.py encoder_encoder_pcn.py" \
  --run "python3 tools/validate_coco64.py"
```
Expected: `BATCHED_RELAXED_STEP nonfinite_layers=[]`, `SHARED_STATE_ALIASES=5`, `GEN_IMG shape=(1, 64, 64, 3) finite=True`, `GEN_TXT shape=(1, 32, 512) finite=True`, `VALIDATE_COCO64_DONE`.

- [ ] **Step 3: Interpret**

All four pass = the 64px config instantiates, batches with finite states, keeps the five shared-latent aliases, and generates in both directions. If `nonfinite_layers` is non-empty or a generation output is non-finite on random weights, note it: it may be genuine instability at the new widths (a real finding, report it) or just untrained-weight behavior over few steps (rerun with more steps to check). If `SHARED_STATE_ALIASES != 5`, the config refactor broke the `share_state_layer` wiring — that is a bug to fix (the CORE constraint).

- [ ] **Step 4: Commit**

```bash
git add tools/validate_coco64.py
git commit -m "validated COCO64_156M on the H200: batches with finite states, keeps the 5 shared-latent aliases, and generates both directions. the bidirectional class now has a working 64px config"
```

---

## Phase 3 exit criteria

A config-driven `EncoderEncoderPCN` where `NATIVE_7B` reproduces the 7.7B model (`GATE_MATCH`) and `COCO64_156M` instantiates near 156M params, runs a batched relaxed step with finite states, preserves the five shared-latent aliases, and generates both directions. Training on COCO-64, the caption data pipeline, generation-quality tuning, and the 64px batch-equivalence re-check are Phase 4, with their own plan.

## Self-Review

- Spec coverage: Section 1 (config-driven constructor) = Tasks 1-3; Section 2 (64px width scheme + initial config) = Task 1's `COCO64_156M` + Task 4 tuning; Section 3 (conv depth kept) = `conv_channels` unchanged in `COCO64_156M` (Task 1 test asserts it); Section 4 (validation: NATIVE_7B GATE_MATCH, COCO64 instantiate/param-count/finite/aliasing/generate) = Tasks 2,3 (GATE_MATCH), 4 (param count), 5 (finite/aliasing/generate); Section 5 (scope) = exit criteria + each task's non-goals. No gaps.
- Placeholder scan: Tasks 2-3 describe a mechanical literal->config substitution rather than showing all ~330 refactored lines — this is intentional and honest for a large order-preserving refactor, and each is anchored by an exact GATE_MATCH command, not prose. The field-mapping is concrete (which literals map to which config fields). All other tasks have complete code and exact commands.
- Type consistency: `PCNConfig` field names used in Tasks 2-5 (`img_resolution`, `conv_channels`, `inter_dim`, `img_dense_relu_widths`, `shared_latent_dims`, `txt_seq_len`, `txt_embed_dim`, `txt_group_widths`, `txt_group_blocks`, `txt_heads`, `txt_sublayers`, `txt_bridge_seq_lens`, `txt_dense_relu_widths`) match the dataclass and both instances in Task 1. `EncoderEncoderPCN(learning_rate, config=...)` signature is consistent across Tasks 2-5. `--config {native7b,coco64}` naming consistent in Task 4.
