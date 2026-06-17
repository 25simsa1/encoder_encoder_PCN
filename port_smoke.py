"""PORT of the verified bidirectional one-energy method onto the 7.7B encoder_encoder_PCN.
PORT_PLAN.md Option C (decided with the user 2026-06-16): conv tower + transformer pyramid as
FEED-FORWARD autodiff edges, only the 5 shared latents free, plus a small decode-to-input anchor
(the model has NO decode-to-pixels/tokens head -- its flatten->Dense heads terminate AT the 5 shared
coupling latents, so feed-forward + cross-only would collapse, violating L1).

Rewrite of the EXECUTION (one energy F, two tapes, tape.gradient). Imports the existing layer
forwards and walks the existing graph wiring purely (no in-place state writes). Architecture and
forward math are untouched.

Runs: build + forward (shapes), then ONE forward+backward at the relaxed states that serves as BOTH
the MEMORY PROBE and the L2 grad-guard source (one backward only -- two eager backwards fragment the
pool and OOM on the 8.26 GB inter9 gradient). 7a checks: relaxation energy descends, states finite,
every module gets a finite nonzero gradient (RAISES on None/zero), one SGD step is takeable. STOP.
"""
import os, sys, time, gc
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.setrecursionlimit(100000)
import numpy as np
import tensorflow as tf
from encoder_encoder_pcn import EncoderEncoderPCN, InputPCNLayer, TransposePCNLayer, FlattenPCNLayer
from dense_pcn_layer import DensePCNLayer
from conv_pcn_layer import Conv2DPCNLayer, MaxPool2DPCNLayer
from transformer_pcn_layer import AttentionPCNLayer, AddNormalizePCNLayer

print("GPUs:", tf.config.list_physical_devices("GPU"))
A_CROSS, A_GEN = 1.0, 2.0
BETA, N_INFER, ALPHA = 0.05, 12, 1e-5
IMG_SHAPE = (1, 572, 572, 3); TXT_SHAPE = (1, 192, 512)
CODE = 16; REC_HW = 64; DEC_SD = 1e-3

t0 = time.time()
m = EncoderEncoderPCN(1e-4)
TXT = [l for l in m.trainable_layers if getattr(l, "share_state_layer", None) is not None]  # dense4,8,12,16,20
IMG = [l.share_state_layer for l in TXT]                                                     # dense2,6,10,14,18
DIMS = [l.num_units for l in IMG]; NS = len(DIMS)
print(f"shared-latent dims: {DIMS}  (NS={NS})   model build {time.time()-t0:.1f}s")

def fwd(layer, xi, xt, memo):
    k = id(layer)
    if k in memo:
        return memo[k]
    if isinstance(layer, InputPCNLayer):
        out = xi if layer is m.img_input else xt
    elif isinstance(layer, AddNormalizePCNLayer):
        out = layer(fwd(layer.prev_layers[0], xi, xt, memo), fwd(layer.prev_layers[1], xi, xt, memo))
    elif isinstance(layer, (DensePCNLayer, Conv2DPCNLayer)):
        out = layer(fwd(layer.prev_layer, xi, xt, memo), set_state=False)
    else:                                                # MaxPool, Flatten, Transpose, Positional, Attention
        out = layer(fwd(layer.prev_layer, xi, xt, memo))
    memo[k] = out
    return out

def taps(xi, xt):
    memo = {}
    return [fwd(l, xi, xt, memo) for l in IMG], [fwd(l, xi, xt, memo) for l in TXT]

# ---------- the small decode-to-input anchor (the ONLY new params) ----------
def Wv(shape, sd=None):
    sd = (1.0 / np.sqrt(int(np.prod(shape[:-1])))) if sd is None else sd
    return tf.Variable(tf.random.normal(shape, stddev=sd), trainable=True)
PROJ = [Wv([DIMS[k], CODE], sd=DEC_SD) for k in range(NS)]
W_DI = Wv([NS * CODE, REC_HW * REC_HW * 3], sd=DEC_SD); B_DI = tf.Variable(tf.zeros(REC_HW * REC_HW * 3))
W_DT = Wv([NS * CODE, TXT_SHAPE[1] * TXT_SHAPE[2]], sd=DEC_SD); B_DT = tf.Variable(tf.zeros(TXT_SHAPE[1] * TXT_SHAPE[2]))
DEC_VARS = PROJ + [W_DI, B_DI, W_DT, B_DT]

def code_of(S):  return tf.concat([tf.nn.relu(S[k] @ PROJ[k]) for k in range(NS)], axis=1)
def dec_img(S):  return tf.nn.sigmoid(code_of(S) @ W_DI + B_DI)
def dec_txt(S):  return code_of(S) @ W_DT + B_DT
def img_target(xi): return tf.reshape(tf.image.resize(xi, [REC_HW, REC_HW]), [tf.shape(xi)[0], -1])
def txt_target(xt): return tf.reshape(xt, [tf.shape(xt)[0], -1])
def se(eps): return tf.reduce_sum(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)
def energy_parts(S, it, tt, xi, xt):
    cross = tf.add_n([se(S[k] - it[k]) + se(S[k] - tt[k]) for k in range(NS)])
    gimg = se(dec_img(S) - img_target(xi)); gtxt = se(dec_txt(S) - txt_target(xt))
    F = 0.5 * tf.reduce_mean(A_CROSS * cross + A_GEN * (gimg + gtxt))
    return F, tf.reduce_mean(gimg), tf.reduce_mean(gtxt)

# ===================== PHASE 0: build + one forward (shapes) =====================
xi = tf.constant(np.random.rand(*IMG_SHAPE).astype("float32"))
xt = tf.constant(np.random.rand(*TXT_SHAPE).astype("float32") * 0.1)
print("\n[Phase 0] one feed-forward (allocates all weights)...")
t1 = time.time()
it0, tt0 = taps(xi, xt)
print(f"  img taps {[tuple(t.shape) for t in it0]}\n  txt taps {[tuple(t.shape) for t in tt0]}  ({time.time()-t1:.1f}s)")
try:
    print(f"  peak after forward: {tf.config.experimental.get_memory_info('GPU:0')['peak']/1e9:.1f} GB")
except Exception as e:
    print("  mem n/a", e)

MW = [getattr(l, a) for l in m.trainable_layers for a in ("wts", "b", "gamma", "beta")
      if isinstance(getattr(l, a, None), tf.Variable)]
ALL_W = MW + DEC_VARS
print(f"  trainable tensors {len(ALL_W)}  (~{sum(int(np.prod(v.shape)) for v in ALL_W)/1e9:.2f}B params; "
      f"decoder adds {sum(int(np.prod(v.shape)) for v in DEC_VARS)/1e6:.1f}M)")

# ===================== 7a: relaxation descent (cheap: reuse taps as constants) =====================
print("\n[7a] relaxation energy descent (S free, both inputs clamped, zero init -> must descend)")
itc = [tf.constant(t) for t in it0]; ttc = [tf.constant(t) for t in tt0]
Sv = [tf.Variable(tf.zeros([1, DIMS[k]])) for k in range(NS)]
Ftraj, GI, GT = [], [], []
for _ in range(N_INFER):
    with tf.GradientTape() as tp:
        for s in Sv: tp.watch(s)
        f, gi, gt = energy_parts(Sv, itc, ttc, xi, xt)
    gs = tp.gradient(f, Sv)
    for s, g in zip(Sv, gs): s.assign_sub(BETA * g)
    Ftraj.append(float(f)); GI.append(float(gi)); GT.append(float(gt))
mono = all(Ftraj[i + 1] <= Ftraj[i] * 1.0001 + 1e-3 for i in range(len(Ftraj) - 1))
F_desc = Ftraj[-1] < Ftraj[0]
states_finite = all(bool(tf.reduce_all(tf.math.is_finite(s))) for s in Sv)
print(f"  F: {Ftraj[0]:.4e} -> {Ftraj[-1]:.4e}   monotone_down={mono}  descends={F_desc}")
print(f"  image-recon err {GI[0]:.3e}->{GI[-1]:.3e}   text-recon err {GT[0]:.3e}->{GT[-1]:.3e}   states_finite={states_finite}")
Srelaxed = [tf.constant(s) for s in Sv]
del itc, ttc, gs, Sv; gc.collect()

# ============ MEMORY PROBE + L2 GRAD GUARD: exactly ONE forward+backward ============
print("\n[MEMORY PROBE + L2 grad guard] one forward + one backward at the relaxed states (batch 1)...")
try:
    tf.config.experimental.reset_memory_stats("GPU:0")
except Exception:
    pass
t2 = time.time()
with tf.GradientTape() as t:
    t.watch(ALL_W)                                       # model wts are trainable=False -> watch explicitly (L2)
    it, tt = taps(xi, xt)                                # FULL forward recomputed INSIDE the tape (L2 silent-freeze guard)
    F, _, _ = energy_parts(Srelaxed, it, tt, xi, xt)
grads = t.gradient(F, ALL_W)
try:
    peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9
except Exception:
    peak = float("nan")
print(f"  F={float(F):.4e}   fwd+bwd {time.time()-t2:.1f}s   PEAK {peak:.1f} GB / 80 GB")

def first(pred): return next(v for v in MW if pred(v))
def gidx(v): return next(i for i, w in enumerate(ALL_W) if w is v)
guard = {
    "conv_tower":     gidx(first(lambda v: len(v.shape) == 4)),
    "transformer":    gidx(first(lambda v: len(v.shape) == 2 and int(v.shape[1]) == 1536)),  # transformer1 kqv (3*512)
    "img_head_dense": gidx(IMG[0].wts), "txt_head_dense": gidx(TXT[0].wts),
    "img_decoder":    gidx(W_DI),       "txt_decoder":    gidx(W_DT),
}
print("  L2 GRAD GUARD (per module):")
guard_ok = True
gnorms = {}
for name, i in guard.items():
    g = grads[i]
    if g is None:
        print(f"    {name:16s} grad=None -> FROZEN MODULE"); guard_ok = False; continue
    gn = float(tf.norm(g)); fin = bool(np.isfinite(gn)); nz = gn > 0; gnorms[name] = gn
    print(f"    {name:16s} |grad|={gn:.3e}  finite={fin} nonzero={nz}")
    guard_ok &= fin and nz
if not guard_ok:
    raise RuntimeError("L2 grad guard FAILED: a module is frozen (None) or has zero/nonfinite gradient.")

# one SGD step is takeable (then free grads); checks weights stay finite
for v, g in zip(ALL_W, grads):
    if g is not None:
        v.assign_sub(ALPHA * g)
wts_finite = all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in (MW[:3] + DEC_VARS))
del grads; gc.collect()
print(f"  one SGD step applied; sampled weights finite = {wts_finite}")

verdict = "7a PASS" if (F_desc and states_finite and guard_ok and wts_finite) else "7a NEEDS REVIEW"
print(f"\n==== {verdict} ====")
print(f"  memory peak (fwd+bwd, batch1) : {peak:.1f} GB / 80 GB  (fits; checkpointing not needed at batch 1)")
print(f"  relaxation energy descends    : {F_desc}  ({Ftraj[0]:.3e}->{Ftraj[-1]:.3e}), monotone={mono}")
print(f"  both recon dirs improve       : image {GI[0]:.2e}->{GI[-1]:.2e}, text {GT[0]:.2e}->{GT[-1]:.2e}")
print(f"  states finite / weights finite: {states_finite} / {wts_finite}")
print(f"  L2 grad guard (all 6 modules) : {guard_ok}")
print(f"  total wall-clock {time.time()-t0:.1f}s")
print("NOTE: only ONE eager backward fits (two fragment the pool -> OOM on the 8.26GB inter9 grad).")
print("      Multiple weight steps need jit_compiled buffer reuse (the PORT_PLAN jit step) -- next up.")
print("      Cross-modal generation (free a raw input through the deep encoder) is a 7c item.")
print("STOP per hard-stop rule (no 7b/7c/7d).")
