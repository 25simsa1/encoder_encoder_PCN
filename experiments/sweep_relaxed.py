"""Validate decoupled inference/learning/bias rates with LARS weight step.
argv: <weight_lr> <num_relax_steps> <mode> <state_lr> <bias_lr> [n_weight_steps]
  mode in {zero, rand}; num_relax_steps==0 -> joint schedule, >0 -> relaxed.
Builds EncoderEncoderPCN(weight_lr), then sets layer.state_lr / layer.bias_lr on every
trainable layer. Logs per WEIGHT step: nonfinite, max|state|, max|wts|, max|b|, #layers
whose |wts| moved >2% off init, min/max trust. Success = weights move AND wts/states/biases
all stay bounded for all weight steps."""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)

weight_lr = float(sys.argv[1]); relax = int(sys.argv[2]); mode = sys.argv[3] if len(sys.argv) > 3 else "zero"
state_lr = float(sys.argv[4]) if len(sys.argv) > 4 else weight_lr
bias_lr = float(sys.argv[5]) if len(sys.argv) > 5 else weight_lr
n_steps = int(sys.argv[6]) if len(sys.argv) > 6 else 15
state_clip = float(sys.argv[7]) if len(sys.argv) > 7 else float('inf')

def val(x): return x.value() if hasattr(x, "value") else x
def nonfinite(t): return bool(tf.reduce_any(tf.math.is_nan(t))) or bool(tf.reduce_any(tf.math.is_inf(t)))
def maxabs(t): return float(tf.reduce_max(tf.abs(t)))

m = None
def per_layer():
    d = {}
    for i, L in enumerate(m.trainable_layers):
        w = getattr(L, "wts", None)
        if w is not None:
            tw = val(w)
            d[i] = (None if nonfinite(tw) else maxabs(tw), getattr(L, "num_units", "-"))
    return d
def agg():
    bad = 0; mxs = 0.0; mxb = 0.0; n_at_clip = 0; n_states = 0
    for L in m.trainable_layers:
        s = getattr(L, "state", None)
        if s is not None:
            n_states += 1
            t = val(s)
            if nonfinite(t): bad += 1
            else:
                ma = maxabs(t); mxs = max(mxs, ma)
                if state_clip != float('inf') and ma >= 0.99 * state_clip: n_at_clip += 1
        b = getattr(L, "b", None)
        if b is not None:
            tb = val(b)
            if not nonfinite(tb): mxb = max(mxb, maxabs(tb))
    return bad, mxs, mxb, n_at_clip, n_states
def n_moved(plw, init, thr=0.02):
    c = 0
    for i, (fw, _) in plw.items():
        iw = init[i][0]
        if fw is not None and iw and abs(fw - iw) / iw > thr:
            c += 1
    return c
def trust_range():
    ts = []
    for L in m.trainable_layers:
        t = getattr(L, "last_trust", None)
        if t is not None:
            f = float(val(t))
            if f == f and f != float("inf"):
                ts.append(f)
    return (min(ts), max(ts)) if ts else (float("nan"), float("nan"))

from encoder_encoder_pcn import EncoderEncoderPCN
if mode == "rand":
    img = tf.random.normal((1, 572, 572, 3)) * 0.1; txt = tf.random.normal((1, 192, 512)) * 0.1
else:
    img = tf.zeros((1, 572, 572, 3)); txt = tf.zeros((1, 192, 512))
mask = tf.zeros((1, 192))

sched = "joint" if relax == 0 else f"relax{relax}"
m = EncoderEncoderPCN(weight_lr)
# decouple the three rates
for layer in m.trainable_layers:
    if hasattr(layer, "state_lr"): layer.state_lr = state_lr
    if hasattr(layer, "bias_lr"):  layer.bias_lr = bias_lr
    if hasattr(layer, "state_clip"): layer.state_clip = state_clip
print(f"### weight_lr={weight_lr:.0e} state_lr={state_lr:.0e} bias_lr={bias_lr:.0e} clip={state_clip} {sched} mode={mode} steps={n_steps}", flush=True)
m.img_input.is_clamped = True; m.txt_input.is_clamped = True
m.pass_through(img, txt, mask)
init = per_layer()
n_wts = len(init)
print(f"init: {n_wts} weight layers, max|wts|={max(v[0] for v in init.values()):.3e}", flush=True)
label = "BOUNDED"; ws_run = 0
for ws in range(1, n_steps + 1):
    m.update_states_wts_b(1) if relax == 0 else m.update_states_wts_b_relaxed(1, relax)
    plw = per_layer(); bad, mxs, mxb, n_clip, n_st = agg()
    mxw = max((v[0] for v in plw.values() if v[0] is not None), default=float("nan"))
    ws_run = ws
    tmin, tmax = trust_range()
    print(f"wstep {ws:2d}: nonfinite={bad} max|state|={mxs:.3e} atclip={n_clip}/{n_st} max|wts|={mxw:.3e} max|b|={mxb:.3e} moved(>2%)={n_moved(plw, init)}/{n_wts} trust[{tmin:.2e},{tmax:.2e}]", flush=True)
    if bad:
        label = "DIVERGED"
        diverged = [(i, w) for i, (fw, w) in plw.items() if fw is None]
        print(f"  first nonfinite weight layers (idx,width): {diverged[:5]}", flush=True)
        break
final = plw
movers = sorted(((abs(fw - init[i][0]) / init[i][0], i, w, init[i][0], fw)
                 for i, (fw, w) in final.items() if fw is not None and init[i][0]), reverse=True)
print(f"  top movers (relchg, idx, width, init->final): {[(round(r,3), i, w, round(a,3), round(b,3)) for r,i,w,a,b in movers[:5]]}", flush=True)
print(f"RESULT weight_lr={weight_lr:.0e} state_lr={state_lr:.0e} bias_lr={bias_lr:.0e} {sched} mode={mode} "
      f"label={label} wsteps={ws_run} moved>2%={n_moved(final, init)}/{n_wts}", flush=True)
