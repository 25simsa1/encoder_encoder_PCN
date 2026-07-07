# Phase 4 SP1: COCO64 Feeder + First Training Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable COCO64 data feeder (64px images + character-level captions) and a training script that overfits ~2000 image-caption pairs with the bidirectional class's relaxed PC schedule, producing a loadable checkpoint whose energy is down/stable (not diverging).

**Architecture:** Reuse the existing 64px image cache directly; encode captions as one-hot character sequences (fixed vocab, length 64) so the model's `txt_embedding` learns the V→512 projection. Train with `update_states_wts_b_relaxed` (relax then one LARS weight step per batch). The class's per-layer beta-less LARS-on-weights is the optimizer; no weight decay, no momentum are added.

**Tech Stack:** TensorFlow 2.21, the Colby H200 via `tools/clusterrun.sh`, `~/tf-env`, pytest for local feeder-logic tests. COCO64 cache at `~/coco_scale/` on the cluster.

## Global Constraints

- Bidirectional class only (`encoder_encoder_pcn.py` + `*_pcn_layer.py`). PC learning only, no functional-version model code, no backprop, no pretrained components (the char vocab + embedding are learned from scratch).
- The five shared-latent pairs stay aliased (`share_state_layer`).
- The optimizer is the class's own per-layer update: beta-less LARS trust-ratio on WEIGHTS only (`w -= learning_rate·(‖w‖/(‖g‖+1e-6))·g`); bias and `state_lr` are plain rates. Add NO weight decay, NO momentum. `state_lr` and the weight LR are separately tunable (all init to the one constructor value); start coupled at 2e-2.
- All GPU runs use `tools/clusterrun.sh` on the H200. Do NOT pass inline `python3 -c` with single quotes to it (its `bash -lc` wrapper mishandles nested single quotes) — use a script file.
- Commits: first-person student voice, NO AI attribution / Co-Authored-By / "Generated with". Commit locally; the controller pushes at checkpoints.
- The character vocabulary is FIXED (defined constant), so V is known and the frozen vocab is identical for SP2.

## File Structure

- Create `coco64_data.py`: the feeder — fixed char vocab, caption encode/decode, mask, image-cache load, batch iterator. One responsibility (COCO64 → model inputs). Imported by training (SP1) and eval (SP2).
- Create `tests/test_coco64_data.py`: local unit tests for the pure encoding logic (no cache, no GPU).
- Modify `pcn_config.py`: `COCO64_156M` `txt_seq_len` 32→64, `txt_embed_dim` 512→50 (=V); add `PCNConfig.__post_init__` length asserts.
- Modify `tests/test_pcn_config.py`: update COCO64 assertions + assert the new field lengths.
- Create `train_coco64.py`: training script — build model, feeder loop, `energy_stats`, checkpointing, energy log.

## The fixed character vocabulary (shared contract)

`CHARS = " abcdefghijklmnopqrstuvwxyz0123456789.,'\"-!?:;()"` (48 characters: space, 26 letters, 10 digits, 11 punctuation). Index 0 = `<pad>`, indices 1..48 = those characters in order, index 49 = `<unk>`. So **V = 50**. Captions are lowercased before encoding; any character not in the set maps to `<unk>`. Sequence length = 64.

---

### Task 1: `coco64_data.py` feeder

**Files:**
- Create: `coco64_data.py`
- Test: `tests/test_coco64_data.py`

**Interfaces:**
- Produces: `VOCAB` (list, len 50), `V=50`, `SEQ=64`; `encode_caption(text:str) -> np.ndarray (64,50) one-hot`; `caption_mask(text:str) -> np.ndarray (64,) additive` (0 at real chars incl. none-beyond-len, -1e9 at pad); `decode(onehot_or_idx) -> str`; `load_batch(n:int, seed:int) -> (img (n,64,64,3) f32, txt (n,64,50) f32, mask (n,64) f32)` reading the cluster cache; `save_vocab(path)`/`load_vocab(path)`.

- [ ] **Step 1: Write the failing local tests**

```python
# tests/test_coco64_data.py
import numpy as np
import coco64_data as D

def test_vocab_size_and_specials():
    assert D.V == 50 and len(D.VOCAB) == 50
    assert D.VOCAB[0] == "<pad>" and D.VOCAB[49] == "<unk>"

def test_encode_shape_and_onehot():
    oh = D.encode_caption("a cat.")
    assert oh.shape == (64, 50)
    assert np.allclose(oh.sum(axis=1)[:6], 1.0)      # first 6 positions are one-hot
    assert np.allclose(oh[6:].sum(axis=1), 1.0)      # pad positions are one-hot on <pad>
    assert oh[6:, 0].all()                           # ...specifically index 0 = <pad>

def test_roundtrip_decode():
    assert D.decode(D.encode_caption("a cat.")) == "a cat."

def test_unknown_char_maps_to_unk():
    oh = D.encode_caption("aéb")                # non-ascii -> <unk>
    assert oh[1, 49] == 1.0                          # position 1 is <unk>

def test_truncation_to_64():
    long = "x" * 100
    oh = D.encode_caption(long)
    assert oh.shape == (64, 50)
    assert D.decode(oh) == "x" * 64

def test_mask_zero_then_negative():
    m = D.caption_mask("a cat.")                     # 6 real chars
    assert (m[:6] == 0.0).all()
    assert (m[6:] < -1e8).all()
```

- [ ] **Step 2: Run, verify fail**

Run: `cd ~/encoder_encoder_PCN && python3 -m pytest tests/test_coco64_data.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'coco64_data'`.

- [ ] **Step 3: Write `coco64_data.py`**

```python
"""COCO64 data feeder: 64px image cache + character-level caption encoding for the
bidirectional class. Fixed vocab (V=50), one-hot char sequences length 64."""
import json
import numpy as np

_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789.,'\"-!?:;()"
VOCAB = ["<pad>"] + list(_CHARS) + ["<unk>"]          # 1 + 48 + 1 = 50
V = len(VOCAB)                                        # 50
SEQ = 64
_C2I = {c: i for i, c in enumerate(VOCAB)}
_PAD, _UNK = 0, V - 1

CACHE = "/home/slsang29/coco_scale"                   # cluster path; images + captions

def _ids(text):
    text = text.lower()[:SEQ]
    return [_C2I.get(c, _UNK) for c in text]

def encode_caption(text):
    ids = _ids(text)
    ids = ids + [_PAD] * (SEQ - len(ids))
    oh = np.zeros((SEQ, V), dtype=np.float32)
    oh[np.arange(SEQ), ids] = 1.0
    return oh

def caption_mask(text):
    n = min(len(text), SEQ)
    m = np.full((SEQ,), -1e9, dtype=np.float32)
    m[:n] = 0.0
    return m

def decode(onehot_or_idx):
    a = np.asarray(onehot_or_idx)
    idx = a.argmax(axis=-1) if a.ndim == 2 else a
    out = []
    for i in idx:
        if i == _PAD:
            break
        out.append(VOCAB[i] if i != _UNK else "?")
    return "".join(out)

def save_vocab(path):
    with open(path, "w") as f:
        json.dump({"vocab": VOCAB, "seq": SEQ}, f)

def load_vocab(path):
    with open(path) as f:
        d = json.load(f)
    assert d["vocab"] == VOCAB and d["seq"] == SEQ, "vocab drift — SP2 must match SP1"
    return VOCAB

def load_batch(n, seed=0, split="train2017"):
    imgs = np.load(f"{CACHE}/imgs_sc_{split}.npy", mmap_mode="r")
    caps = open(f"{CACHE}/caps_sc_{split}.txt").read().splitlines()
    k = min(n, imgs.shape[0], len(caps))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(k)                          # fixed 2k subset = first k, shuffled
    img = np.asarray(imgs[:k], dtype=np.float32)[idx]
    txt = np.stack([encode_caption(caps[i]) for i in range(k)])[idx]
    mask = np.stack([caption_mask(caps[i]) for i in range(k)])[idx]
    return img, txt, mask
```

- [ ] **Step 4: Run local tests, verify pass**

Run: `python3 -m pytest tests/test_coco64_data.py -q`
Expected: PASS (6 tests). (These do not touch the cache or GPU.)

- [ ] **Step 5: Cluster cache smoke** — confirm `load_batch` reads the real cache.

Create `tools/_coco_feed_smoke.py`:
```python
import coco64_data as D
img, txt, mask = D.load_batch(8, seed=0)
print("FEED_OK", img.shape, img.dtype, float(img.min()), float(img.max()), txt.shape, mask.shape,
      "sample:", repr(D.decode(txt[0])))
```
Run (Bash timeout 600000 ms): `tools/clusterrun.sh --name coco_feed --gpu H200 --mem 16G --cpus 2 --time 00:10:00 --sync "coco64_data.py tools/_coco_feed_smoke.py" --run "python3 tools/_coco_feed_smoke.py"`
Expected: `FEED_OK (8,64,64,3) float32 0.0..1.0 (8,64,50) (8,64) sample: '<some caption text>'`. Delete `tools/_coco_feed_smoke.py` after (do not commit it).

- [ ] **Step 6: Commit**

```bash
git add coco64_data.py tests/test_coco64_data.py
git commit -m "added the coco64 feeder: reuse the 64px image cache and encode captions char-level (fixed V=50 vocab, one-hot seq 64) with an additive pad mask + decode for later eval"
```

---

### Task 2: config update for character-level text + config asserts

**Files:**
- Modify: `pcn_config.py` (`COCO64_156M` + `PCNConfig.__post_init__`)
- Modify: `tests/test_pcn_config.py`

**Interfaces:**
- Consumes: `V=50`, `SEQ=64` from `coco64_data` (as literals; no import, to avoid a config→data dependency).
- Produces: `COCO64_156M` with `txt_seq_len=64`, `txt_embed_dim=50`.

- [ ] **Step 1: Update the config test (failing)**

In `tests/test_pcn_config.py`, change the COCO64 assertions to `txt_seq_len == 64` and `txt_embed_dim == 50`, and add:
```python
def test_config_post_init_lengths():
    from pcn_config import PCNConfig, NATIVE_7B, COCO64_156M
    for c in (NATIVE_7B, COCO64_156M):
        assert len(c.conv_channels) == 9
        assert len(c.img_dense_relu_widths) == 5 == len(c.shared_latent_dims) == len(c.txt_dense_relu_widths) == len(c.txt_tap_indices)
        assert len(c.txt_group_widths) == len(c.txt_group_blocks)
        assert len(c.txt_bridge_seq_lens) == len(c.txt_group_widths) - 1
```

- [ ] **Step 2: Run, verify fail**

Run: `python3 -m pytest tests/test_pcn_config.py -q`
Expected: FAIL (COCO64 still has `txt_seq_len=32`/`txt_embed_dim=512`; no `__post_init__`).

- [ ] **Step 3: Edit `pcn_config.py`**

In `COCO64_156M`, set `txt_seq_len=64` and `txt_embed_dim=50` (add a comment: `# 50 = coco64_data.V (one-hot char), 64 = seq`). Add to `PCNConfig`:
```python
    def __post_init__(self):
        assert len(self.conv_channels) == 9, "expected 9 conv channels"
        assert len(self.img_dense_relu_widths) == len(self.shared_latent_dims) == len(self.txt_dense_relu_widths) == len(self.txt_tap_indices) == 5, "expected 5 per-scale values"
        assert len(self.txt_group_widths) == len(self.txt_group_blocks), "group widths/blocks length mismatch"
        assert len(self.txt_bridge_seq_lens) == len(self.txt_group_widths) - 1, "expected one bridge per group gap"
```
(Frozen dataclasses support `__post_init__`; it runs on the existing instances at import.)

- [ ] **Step 4: Run config test, verify pass**

Run: `python3 -m pytest tests/test_pcn_config.py -q`
Expected: PASS.

- [ ] **Step 5: Re-verify COCO64 builds + param count on the H200**

Run (Bash timeout 600000 ms): `tools/clusterrun.sh --name p4_count --gpu H200 --mem 48G --cpus 4 --time 00:15:00 --sync "pcn_config.py encoder_encoder_pcn.py conv_pcn_layer.py tools/count_params.py coco64_data.py" --run "python3 tools/count_params.py --config coco64"`
Expected: a `TOTAL_PARAMS=... (XXX.XM)` line in [125M, 190M]. NOTE: `count_params.py` feeds `(1, seq, txt_embed_dim)` — with `txt_embed_dim=50` now, this exercises the real char-input dim. If the count drifts out of band, adjust ONLY the COCO64 `img_dense_relu_widths`/`shared_latent_dims` (cheapest first) and re-run. Record the final count.

- [ ] **Step 6: Commit**

```bash
git add pcn_config.py tests/test_pcn_config.py
git commit -m "pointed COCO64 at char-level text (seq 64, embed dim 50 = the one-hot vocab) and added PCNConfig length asserts so malformed capacity configs fail loudly. re-verified it still builds near 156M"
```

---

### Task 3: `train_coco64.py` training script + smoke

**Files:**
- Create: `train_coco64.py`
- Uses: `coco64_data` (Task 1), `EncoderEncoderPCN`/`COCO64_156M`.

**Interfaces:**
- Produces: a CLI `python3 train_coco64.py --pairs N --epochs E --lr LR --relax R --batch B --ckpt DIR [--energy-every K] [--resume]` that trains and checkpoints; prints `[step ...] energy=<e> max_abs_state=<m>` lines and `TRAIN_DONE`.

- [ ] **Step 1: Write `train_coco64.py`**

```python
"""Overfit the bidirectional class on a COCO64 subset with its relaxed PC schedule.
The optimizer is the class's own beta-less LARS-on-weights; this script only sets the
learning rate and runs the schedule. Logs a PC-energy proxy + max|state| for divergence."""
import os, argparse, time
import tensorflow as tf, numpy as np
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import COCO64_156M
import coco64_data as D

def energy_stats(m):
    """max|state| (divergence) + mean per-layer forward prediction error (PC energy proxy)."""
    max_abs, total_err, n = 0.0, 0.0, 0
    for L in m.trainable_layers:
        s = getattr(L, "state", None)
        if s is None:
            continue
        max_abs = max(max_abs, float(tf.reduce_max(tf.abs(s))))
        prev = getattr(L, "prev_layer", None)
        if prev is not None:
            try:
                err = float(tf.reduce_mean(tf.square(L(prev.predict_next()) - L.predict_next())))
                total_err += err; n += 1
            except Exception:
                pass
    return total_err / max(1, n), max_abs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=2000); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-2); ap.add_argument("--relax", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--ckpt", default="ckpt_coco64")
    ap.add_argument("--energy-every", type=int, default=50); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    img, txt, mask = D.load_batch(a.pairs, seed=0)
    print(f"data: img{img.shape} txt{txt.shape} mask{mask.shape}", flush=True)
    m = EncoderEncoderPCN(a.lr, config=COCO64_156M)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    # realize weights so they can be checkpointed
    b0 = slice(0, a.batch)
    m.pass_through(tf.convert_to_tensor(img[b0]), tf.convert_to_tensor(txt[b0]), tf.convert_to_tensor(mask[b0]))
    ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts", "b", "gamma", "beta")
             if isinstance(getattr(l, at, None), tf.Variable)]
    ckpt = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    mgr = tf.train.CheckpointManager(ckpt, a.ckpt, max_to_keep=1)
    if a.resume and mgr.latest_checkpoint:
        ckpt.restore(mgr.latest_checkpoint); print("resumed", mgr.latest_checkpoint, flush=True)

    N = img.shape[0]; step = 0; t0 = time.time()
    for ep in range(a.epochs):
        order = np.random.default_rng(ep).permutation(N)
        for s in range(0, N - a.batch + 1, a.batch):
            bi = order[s:s + a.batch]
            m.img_input.is_clamped = True; m.txt_input.is_clamped = True
            m.pass_through(tf.convert_to_tensor(img[bi]), tf.convert_to_tensor(txt[bi]), tf.convert_to_tensor(mask[bi]))
            m.update_states_wts_b_relaxed(num_weight_steps=1, num_relax_steps=a.relax)
            step += 1
            if step % a.energy_every == 0:
                e, mx = energy_stats(m)
                print(f"[step {step} ep {ep}] energy={e:.5f} max_abs_state={mx:.3f} ({(time.time()-t0)/step:.2f}s/step)", flush=True)
                if not np.isfinite(e) or not np.isfinite(mx) or mx > 1e6:
                    print(f"DIVERGED at step {step}", flush=True); mgr.save(); return
            if step % 1000 == 0:
                mgr.save(); print(f"ckpt @ {step}", flush=True)
    mgr.save(); print("TRAIN_DONE", flush=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke on the H200** — a few steps, energy finite + moving, checkpoint saves and reloads.

Run (Bash timeout 600000 ms): `tools/clusterrun.sh --name p4_train_smoke --gpu H200 --mem 64G --cpus 4 --time 00:25:00 --sync "train_coco64.py coco64_data.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py" --run "python3 train_coco64.py --pairs 64 --epochs 3 --batch 8 --relax 15 --energy-every 5 --ckpt ckpt_smoke && python3 train_coco64.py --pairs 64 --epochs 1 --batch 8 --relax 15 --ckpt ckpt_smoke --resume"`
Expected: several `[step ...] energy=<finite> max_abs_state=<finite>` lines with energy generally decreasing (or at least finite/bounded), a `ckpt` save, `TRAIN_DONE`, then a clean `resumed .../ckpt_smoke-...` on the second invocation. No `DIVERGED`, no NaN. (Uses 64 pairs so the smoke is fast; the real run is Task 4.)

- [ ] **Step 3: Commit**

```bash
git add train_coco64.py
git commit -m "training script for the coco64 overfit: relaxed PC schedule (relax then one LARS weight step per batch), energy + max|state| logging with a divergence guard, and tf.train.Checkpoint save/resume over the class weights"
```

---

### Task 4: the 2k overfit run (deliverable)

**Files:** none (runs Task 3's script). Produces `ckpt_coco64/` on the cluster + a training log.

- [ ] **Step 1: Launch the overfit run on the H200.**

Run (Bash timeout 600000 ms; ~200 epochs × 2000/8 = 50k steps; if that exceeds the wall clock, the checkpoint-every-1000 + `--resume` lets it continue in a second job): `tools/clusterrun.sh --name p4_overfit --gpu H200 --mem 96G --cpus 4 --time 08:00:00 --sync "train_coco64.py coco64_data.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py" --run "python3 train_coco64.py --pairs 2000 --epochs 200 --lr 2e-2 --relax 15 --batch 8 --energy-every 50 --ckpt ckpt_coco64"`
Expected: a stream of `[step ...] energy=... max_abs_state=...` with the energy trending DOWN (the model fitting the 2k set) and `max_abs_state` bounded, periodic `ckpt @ N`, and `TRAIN_DONE`. If `DIVERGED` appears, that is the finding to report — do NOT force it; the divergence levers (in order) are: re-run with `--lr 1e-2`, then a short LR ramp (needs a small script change — escalate if reached), then lower `state_lr`, then `state_clip`. Record which lever, if any.

- [ ] **Step 2: Confirm the checkpoint loads.**

Create `tools/_ckpt_check.py`:
```python
import tensorflow as tf
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import COCO64_156M
import coco64_data as D
m = EncoderEncoderPCN(2e-2, config=COCO64_156M)
img, txt, mask = D.load_batch(8, seed=0)
m.img_input.is_clamped = True; m.txt_input.is_clamped = True
m.pass_through(tf.convert_to_tensor(img), tf.convert_to_tensor(txt), tf.convert_to_tensor(mask))
ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts","b","gamma","beta") if isinstance(getattr(l, at, None), tf.Variable)]
ckpt = tf.train.Checkpoint(**{f"v{i}": v for i,v in enumerate(ALL_W)})
st = ckpt.restore(tf.train.latest_checkpoint("ckpt_coco64")).expect_partial()
print("CKPT_LOAD_OK nweights=", len(ALL_W))
```
Run (Bash timeout 600000 ms): `tools/clusterrun.sh --name p4_ckpt --gpu H200 --mem 48G --cpus 4 --time 00:15:00 --sync "tools/_ckpt_check.py coco64_data.py encoder_encoder_pcn.py pcn_config.py conv_pcn_layer.py transformer_pcn_layer.py dense_pcn_layer.py" --run "python3 tools/_ckpt_check.py"`
Expected: `CKPT_LOAD_OK nweights=<N>`. Delete `tools/_ckpt_check.py` after (do not commit).

- [ ] **Step 3: Record the outcome** (final energy, steps, whether any divergence lever was used) in the SP1 report. No code commit; the deliverable is the checkpoint + log. SP1 is complete when the run finished without unresolved divergence, energy is down/stable, and the checkpoint loads.

---

## SP1 exit criteria

A reusable `coco64_data` feeder (local-tested + cluster-smoked), `COCO64_156M` pointed at char-level text and re-verified near 156M, a `train_coco64.py` that trains + checkpoints, and a completed 2k overfit run whose energy is down/stable with a loadable checkpoint. Recognizable generation and all metrics are SP2.

## Self-Review

- Spec coverage: feeder (image reuse + char one-hot + mask + decode + frozen vocab) = Task 1; config update (seq 64, embed 50, asserts) = Task 2; training (relaxed schedule, LARS-only, no WD/momentum, checkpoint, energy log, divergence levers) = Task 3; the 2k overfit run + checkpoint-loads = Task 4. The `state_lr`-decoupled note is honored (script sets one LR; levers mention lowering `state_lr`). No spec requirement unmapped.
- Placeholder scan: all code is complete; the two throwaway probes (`_coco_feed_smoke.py`, `_ckpt_check.py`) are shown in full and deleted, not committed. "many epochs" is concretized as `--epochs 200` with checkpoint/resume for wall-clock. Divergence handling names concrete levers rather than "handle errors."
- Type consistency: `encode_caption`/`caption_mask`/`decode`/`load_batch`/`V=50`/`SEQ=64` names match between `coco64_data.py` (Task 1), the config values (Task 2, embed_dim=50/seq 64), and `train_coco64.py` (Task 3). `energy_stats` returns `(energy, max_abs)` and is unpacked that way. Checkpoint gathering (`wts/b/gamma/beta`) matches between Task 3 and Task 4's loader.

---

### Task 2 addendum (discovered 2026-07-07): txt_embedding width coupling

SP1.2 is the first config to decouple `txt_embed_dim` (now the char-vocab input dim, 50) from `txt_group_widths[0]` (the first transformer width, 512). `encoder_encoder_pcn.py:272,275` build `txt_embedding`/`pos_encoding` at `config.txt_embed_dim`, so the first transformer block's residual add fails (50 vs 512) for COCO64. They were always equal before, so this was latent. Fix, folded into Task 2: change lines 272 and 275 to `config.txt_group_widths[0]` (the transformer input width). This is a no-op for NATIVE_7B (512==512), so `NATIVE_7B` must still `GATE_MATCH nlayers=143` — that becomes Task 2's gate 1, in addition to the COCO64 param-count. After the fix, `txt_embed_dim` is a tooling/feeder-only field (the one-hot input dim); it must not otherwise appear in the model.
