"""Validate the relaxed PC training schedule (update_states_wts_b_relaxed).
argv: <lr> <num_relax_steps> <mode>   mode in {zero, rand}.
Fresh model, inputs clamped + forward, then 15 weight steps each preceded by
num_relax_steps state-relaxation sweeps. Logs after every WEIGHT step.
Success = weights move off ~1.0 init AND stay bounded; frozen-but-stable = FAIL."""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

lr = float(sys.argv[1]); relax = int(sys.argv[2]); mode = sys.argv[3] if len(sys.argv) > 3 else "zero"

def val(x): return x.value() if hasattr(x, "value") else x
def nonfinite(t): return bool(tf.reduce_any(tf.math.is_nan(t))) or bool(tf.reduce_any(tf.math.is_inf(t)))
def maxabs(t): return float(tf.reduce_max(tf.abs(t)))

m = None
def scan():
    bad = 0; mxs = 0.0; mxw = 0.0
    for L in m.trainable_layers:
        s = getattr(L, "state", None)
        if s is not None:
            t = val(s)
            if nonfinite(t): bad += 1
            else: mxs = max(mxs, maxabs(t))
        w = getattr(L, "wts", None)
        if w is not None:
            tw = val(w)
            if nonfinite(tw): bad += 1
            else: mxw = max(mxw, maxabs(tw))
    return bad, mxs, mxw

from encoder_encoder_pcn import EncoderEncoderPCN
if mode == "rand":
    img = tf.random.normal((1, 572, 572, 3)) * 0.1; txt = tf.random.normal((1, 192, 512)) * 0.1
else:
    img = tf.zeros((1, 572, 572, 3)); txt = tf.zeros((1, 192, 512))
mask = tf.zeros((1, 192))

m = EncoderEncoderPCN(lr)
print(f"### lr={lr:.0e} relax={relax} mode={mode}", flush=True)
m.img_input.is_clamped = True; m.txt_input.is_clamped = True
m.pass_through(img, txt, mask)
_, mxs0, mxw0 = scan()
print(f"init(post-forward): max|state|={mxs0:.3e} max|wts|={mxw0:.3e}", flush=True)
label = "BOUNDED"; ws_run = 0; last_s = mxs0; last_w = mxw0
for ws in range(1, 16):
    m.update_states_wts_b_relaxed(1, relax)   # relax states, then ONE weight/bias step
    bad, mxs, mxw = scan()
    ws_run = ws
    print(f"wstep {ws:2d}: nonfinite={bad} max|state|={mxs:.3e} max|wts|={mxw:.3e}", flush=True)
    if bad:
        label = "DIVERGED"; break
    last_s, last_w = mxs, mxw
moved = (last_w > 1.25 * mxw0) or (last_w < 0.80 * mxw0)
print(f"RESULT lr={lr:.0e} relax={relax} mode={mode} label={label} wsteps={ws_run} "
      f"init_max_wts={mxw0:.3e} final_max_wts={last_w:.3e} final_max_state={last_s:.3e} "
      f"weights_moved={moved}", flush=True)
