"""Long-horizon (up to 300 weight-step) durability + energy trace for the best
stable regime. Driver only -- uses the existing per-layer update methods; no model
changes. Best combo: weight_lr=1e-1, state_lr=1e-4, bias_lr=0 (frozen), state_clip=30,
relax=32, small random inputs.

Logs at step 1 and every 25 steps (streamed):
  max|wts|, max|conv weight|, max|state|, max|conv state|, #layers moved off init,
  and a prediction-error ENERGY proxy measured AFTER the relaxation phase:
    energy = sum_layers reduce_mean((prev.predict_next() - layer.predict_prev())**2)

Answers: (1) does max|conv state| come off the 30 ceiling or stay pinned? (2) do conv
weights plateau or keep creeping? plus energy trend. Early-stops ~150 if the verdict
(load-bearing clamp, no learning) is already in."""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gc
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from conv_pcn_layer import Conv2DPCNLayer
from encoder_encoder_pcn import EncoderEncoderPCN

WEIGHT_LR = 1e-1; STATE_LR = 1e-4; BIAS_LR = 0.0; CLIP = 30.0; RELAX = 32; MAXSTEPS = 300

def val(x): return x.value() if hasattr(x, "value") else x
def nonfinite(t): return bool(tf.reduce_any(tf.math.is_nan(t))) or bool(tf.reduce_any(tf.math.is_inf(t)))
def maxabs(t): return float(tf.reduce_max(tf.abs(t)))
def is_conv(L): return isinstance(L, Conv2DPCNLayer)

img = tf.random.normal((1, 572, 572, 3)) * 0.1
txt = tf.random.normal((1, 192, 512)) * 0.1
mask = tf.zeros((1, 192))

m = EncoderEncoderPCN(WEIGHT_LR)
for L in m.trainable_layers:
    if hasattr(L, "state_lr"): L.state_lr = STATE_LR
    if hasattr(L, "bias_lr"): L.bias_lr = BIAS_LR
    if hasattr(L, "state_clip"): L.state_clip = CLIP
m.img_input.is_clamped = True; m.txt_input.is_clamped = True
m.pass_through(img, txt, mask)

init = {}
for i, L in enumerate(m.trainable_layers):
    w = getattr(L, "wts", None)
    if w is not None and not nonfinite(val(w)): init[i] = maxabs(val(w))
n_wts = len(init)

def metrics():
    mxw = mxcw = mxs = mxcs = 0.0; bad = 0; moved = 0
    for i, L in enumerate(m.trainable_layers):
        w = getattr(L, "wts", None)
        if w is not None:
            tw = val(w)
            if nonfinite(tw): bad += 1
            else:
                a = maxabs(tw); mxw = max(mxw, a)
                if is_conv(L): mxcw = max(mxcw, a)
                if i in init and init[i] and abs(a - init[i]) / init[i] > 0.02: moved += 1
        s = getattr(L, "state", None)
        if s is not None:
            ts = val(s)
            if nonfinite(ts): bad += 1
            else:
                a = maxabs(ts); mxs = max(mxs, a)
                if is_conv(L): mxcs = max(mxcs, a)
    return bad, mxw, mxcw, mxs, mxcs, moved

def energy():
    e = 0.0
    for L in m.trainable_layers:
        prev = getattr(L, "prev_layer", None)
        if prev is None: continue
        try:
            d = prev.predict_next() - L.predict_prev()
            e += float(tf.reduce_mean(d * d))
        except Exception:
            pass
    return e

log_steps = set([1] + list(range(25, MAXSTEPS + 1, 25)))
print(f"### 300-step durability  wlr={WEIGHT_LR} slr={STATE_LR} blr={BIAS_LR} clip={CLIP} relax={RELAX} inputs=randn*0.1 n_wts={n_wts}", flush=True)
prev_mxcw = -1.0; prev_en = -1.0
for ws in range(1, MAXSTEPS + 1):
    for _ in range(RELAX):
        for L in m.trainable_layers:
            L.update_state()
    en = energy() if ws in log_steps else None
    for L in m.trainable_layers:
        L.update_wts(); L.update_b()
    gc.collect()
    if ws in log_steps:
        bad, mxw, mxcw, mxs, mxcs, moved = metrics()
        print(f"wstep {ws:3d}: nf={bad} max|wts|={mxw:.3e} max|convW|={mxcw:.3e} max|state|={mxs:.3e} max|convS|={mxcs:.3e} moved={moved}/{n_wts} energy={en:.4e}", flush=True)
        if bad:
            print("  DIVERGED", flush=True); break
        pinned = mxcs >= 0.99 * CLIP
        climbing = prev_mxcw > 0 and mxcw > prev_mxcw * 1.005
        en_flat_up = prev_en > 0 and en >= prev_en * 0.98
        if ws >= 150 and pinned and climbing and en_flat_up:
            print(f"  EARLY_STOP @ {ws}: conv states pinned at clip + conv weights still climbing + energy flat/rising = load-bearing clamp, no learning", flush=True)
            break
        prev_mxcw = mxcw; prev_en = en
print("DONE", flush=True)
