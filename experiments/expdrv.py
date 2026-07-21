"""Energy-trend experiment driver. argv: <relax> <state_clip> <n_steps> [log_every=10]
Best combo: weight_lr=1e-1, state_lr=1e-4, bias_lr=0 (frozen), small random inputs.
Logs at step 1 and every log_every weight steps: energy (after relaxation), max|state|,
atclip (state layers at the clip ceiling), max|wts|, #layers moved off init.
PASS = energy TRENDING DOWN with the clip NOT load-bearing (atclip small/zero)."""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gc
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from conv_pcn_layer import Conv2DPCNLayer
from encoder_encoder_pcn import EncoderEncoderPCN

RELAX = int(sys.argv[1]); CLIP = float(sys.argv[2]); NSTEPS = int(sys.argv[3])
LOG = int(sys.argv[4]) if len(sys.argv) > 4 else 10
WLR = 1e-1; SLR = 1e-4; BLR = 0.0

def val(x): return x.value() if hasattr(x, "value") else x
def nonfinite(t): return bool(tf.reduce_any(tf.math.is_nan(t))) or bool(tf.reduce_any(tf.math.is_inf(t)))
def maxabs(t): return float(tf.reduce_max(tf.abs(t)))

img = tf.random.normal((1, 572, 572, 3)) * 0.1
txt = tf.random.normal((1, 192, 512)) * 0.1
mask = tf.zeros((1, 192))

m = EncoderEncoderPCN(WLR)
for L in m.trainable_layers:
    if hasattr(L, "state_lr"): L.state_lr = SLR
    if hasattr(L, "bias_lr"): L.bias_lr = BLR
    if hasattr(L, "state_clip"): L.state_clip = CLIP
m.img_input.is_clamped = True; m.txt_input.is_clamped = True
m.pass_through(img, txt, mask)
init = {}
for i, L in enumerate(m.trainable_layers):
    w = getattr(L, "wts", None)
    if w is not None and not nonfinite(val(w)): init[i] = maxabs(val(w))
n_wts = len(init)

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

def metrics():
    mxw = mxs = 0.0; bad = 0; moved = 0; n_at_clip = 0; n_states = 0
    for i, L in enumerate(m.trainable_layers):
        w = getattr(L, "wts", None)
        if w is not None:
            tw = val(w)
            if nonfinite(tw): bad += 1
            else:
                a = maxabs(tw); mxw = max(mxw, a)
                if i in init and init[i] and abs(a - init[i]) / init[i] > 0.02: moved += 1
        s = getattr(L, "state", None)
        if s is not None:
            n_states += 1
            ts = val(s)
            if nonfinite(ts): bad += 1
            else:
                a = maxabs(ts); mxs = max(mxs, a)
                if CLIP != float('inf') and a >= 0.99 * CLIP: n_at_clip += 1
    return bad, mxw, mxs, moved, n_at_clip, n_states

log_steps = set([1] + list(range(LOG, NSTEPS + 1, LOG)))
print(f"### EXP relax={RELAX} clip={CLIP} nsteps={NSTEPS} wlr={WLR} slr={SLR} blr={BLR} inputs=randn*0.1 n_wts={n_wts}", flush=True)
for ws in range(1, NSTEPS + 1):
    for _ in range(RELAX):
        for L in m.trainable_layers:
            L.update_state()
    en = energy() if ws in log_steps else None
    for L in m.trainable_layers:
        L.update_wts(); L.update_b()
    gc.collect()
    if ws in log_steps:
        bad, mxw, mxs, moved, nclip, nst = metrics()
        print(f"wstep {ws:3d}: nf={bad} energy={en:.4e} max|state|={mxs:.3e} atclip={nclip}/{nst} max|wts|={mxw:.3e} moved={moved}/{n_wts}", flush=True)
        if bad:
            print("  DIVERGED", flush=True); break
print("DONE", flush=True)
