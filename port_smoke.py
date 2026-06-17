"""PORT of the bidirectional one-energy method onto the 7.7B encoder_encoder_PCN (Option C).
PRE-7b CONDITIONING FIX: the per-edge MEAN-normalized energy (committed b00396b) put gradients at
order 1 but still ~5700x imbalanced across modules, so one global SGD lr froze (1e-5) or exploded
(1e-4). This version applies the per-variable LARS trust step on the SAME normalized energy, which
equalizes the RELATIVE step per module regardless of gradient-magnitude spread (the fix validated in
the miniatures). The weight update is the ONLY change here; F, the architecture, and the forward math
are unchanged. Graph-mode @tf.function (XLA jit_compile was too slow on the 7.76B graph),
cuda_malloc_async. The sweep does PROPER relax-then-step (re-relax S each weight step), not fixed S0.

LARS step (per trainable variable):  trust = ||var|| / (||g|| + 1e-6);  var -= lr * trust * g
so each variable's update has norm lr*||var|| -> a uniform RELATIVE step ~lr regardless of ||g||.
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
REL_C, N_INFER = 0.05, 8
LRS = [1e-3, 1e-2, 1e-1, 1.0]; N_W = 22
IMG_SHAPE = (1, 572, 572, 3); TXT_SHAPE = (1, 192, 512)
CODE = 16; REC_HW = 64; DEC_SD = 1e-3

t0 = time.time()
m = EncoderEncoderPCN(1e-4)
TXT = [l for l in m.trainable_layers if getattr(l, "share_state_layer", None) is not None]
IMG = [l.share_state_layer for l in TXT]
DIMS = [l.num_units for l in IMG]; NS = len(DIMS); betas = [REL_C * d for d in DIMS]
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
    else:
        out = layer(fwd(layer.prev_layer, xi, xt, memo))
    memo[k] = out
    return out
def taps(xi, xt):
    memo = {}
    return [fwd(l, xi, xt, memo) for l in IMG], [fwd(l, xi, xt, memo) for l in TXT]

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
def mse(eps):    return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)   # per-element MEAN per edge
def F_of(S, it, tt):
    cross = tf.add_n([mse(S[k] - it[k]) + mse(S[k] - tt[k]) for k in range(NS)])
    return 0.5 * tf.reduce_mean(A_CROSS * cross + A_GEN * (mse(dec_img(S) - IMG_T) + mse(dec_txt(S) - TXT_T)))

# ===================== build + forward =====================
xi = tf.constant(np.random.rand(*IMG_SHAPE).astype("float32"))
xt = tf.constant(np.random.rand(*TXT_SHAPE).astype("float32") * 0.1)
it0, tt0 = taps(xi, xt)
print(f"[Phase 0] tap shapes ok: {[tuple(t.shape) for t in it0]}")
MW = [getattr(l, a) for l in m.trainable_layers for a in ("wts", "b", "gamma", "beta")
      if isinstance(getattr(l, a, None), tf.Variable)]
ALL_W = MW + DEC_VARS
print(f"  trainable tensors {len(ALL_W)}  (~{sum(int(np.prod(v.shape)) for v in ALL_W)/1e9:.2f}B params)")
IMG_T = tf.constant(tf.reshape(tf.image.resize(xi, [REC_HW, REC_HW]), [1, -1]))
TXT_T = tf.constant(tf.reshape(xt, [1, -1]))

@tf.function
def get_taps(xi, xt):
    return taps(xi, xt)

def relax_S(S, it, tt, n):                                # eager, cheap (S small, taps constant)
    Sv = list(S)
    for _ in range(n):
        with tf.GradientTape() as tp:
            tp.watch(Sv)
            f = F_of(Sv, it, tt)
        g = tp.gradient(f, Sv)
        Sv = [Sv[k] - betas[k] * g[k] for k in range(NS)]
    return Sv

@tf.function                                              # graph mode: buffers reuse across calls -> multi-step fits
def weight_step(xi, xt, S, lr):
    with tf.GradientTape() as t:
        t.watch(ALL_W)
        it, tt = taps(xi, xt)                             # FULL forward inside the tape (L2)
        F = F_of(S, it, tt)
    g = t.gradient(F, ALL_W)
    trusts = []
    for v, gg in zip(ALL_W, g):
        if gg is None:
            continue
        tr = tf.norm(v) / (tf.norm(gg) + 1e-6)            # LARS trust ratio (per variable)
        trusts.append(tr)
        v.assign_sub(lr * tr * gg)                        # uniform RELATIVE step ~lr regardless of ||g||
    trs = tf.stack(trusts)
    mx = tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
    return F, mx, tf.reduce_min(trs), tf.reduce_max(trs)

# ===================== initial relaxation descent (zero init) =====================
it0c = [tf.constant(t) for t in it0]; tt0c = [tf.constant(t) for t in tt0]
Sv = [tf.zeros([1, DIMS[k]]) for k in range(NS)]; Ft = []
for _ in range(12):
    with tf.GradientTape() as tp:
        tp.watch(Sv); f = F_of(Sv, it0c, tt0c)
    g = tp.gradient(f, Sv); Sv = [Sv[k] - betas[k] * g[k] for k in range(NS)]; Ft.append(float(f))
S0 = [tf.constant(s) for s in Sv]
print(f"[relaxation] F {Ft[0]:.4e} -> {Ft[-1]:.4e}  descends={Ft[-1]<Ft[0]}  states_finite={all(bool(tf.reduce_all(tf.math.is_finite(s))) for s in S0)}")
del it0c, tt0c, Sv, g; gc.collect()

# ===================== LARS sweep: proper relax-then-step =====================
print(f"\n[LARS sweep] relax-then-step ({N_W} weight steps/lr, re-relax {N_INFER} each); MOVE meaningfully AND BOUNDED?")
cw = next(v for v in MW if len(v.shape) == 4)
try:
    tf.config.experimental.reset_memory_stats("GPU:0")
except Exception:
    pass
peak = float("nan"); results = []; S = S0
for lr in LRS:
    cw0 = tf.identity(cw); lr_t = tf.constant(lr, tf.float32)
    Fs, mxs, trmin, trmax = [], [], [], []
    for step in range(N_W):
        it, tt = get_taps(xi, xt)
        S = relax_S([tf.constant(s) for s in S], it, tt, N_INFER)      # re-relax under current weights
        F, mx, tmn, tmx = weight_step(xi, xt, tuple(tf.constant(s) for s in S), lr_t)
        Fs.append(float(F)); mxs.append(float(mx)); trmin.append(float(tmn)); trmax.append(float(tmx))
        if np.isnan(mxs[-1]) or mxs[-1] > 1e4:
            break
    if np.isnan(peak):
        try: peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9
        except Exception: pass
    seg_moved = float(tf.norm(cw - cw0) / (tf.norm(cw0) + 1e-9))
    bounded = all(np.isfinite(v) and v < 1e3 for v in mxs)
    F_down = (len(Fs) >= 2) and (np.mean(Fs[-3:]) < np.mean(Fs[:3]))
    results.append((lr, Fs[0], Fs[-1], seg_moved, bounded, min(trmin), max(trmax), F_down, len(Fs)))
    print(f"  lr={lr:.0e}: F {Fs[0]:.3e}->{Fs[-1]:.3e} F_down={F_down}  conv_moved={seg_moved:.2e}  bounded={bounded}  "
          f"trust[{min(trmin):.1e},{max(trmax):.1e}]  steps={len(Fs)}")

moving_and_bounded = [r[0] for r in results if r[3] > 1e-3 and r[4]]
largest = max(moving_and_bounded) if moving_and_bounded else None
learns = [r[0] for r in results if r[3] > 1e-3 and r[4] and r[7]]    # moved + bounded + F trending down

print("\n==== PRE-7b LARS CONDITIONING SUMMARY ====")
print(f"  memory peak (graph-mode, batch1): {peak:.1f} GB / 80 GB")
print(f"  per lr: " + " | ".join(f"{r[0]:.0e}:moved={r[3]:.1e},bounded={r[4]},Fdown={r[7]}" for r in results))
glob_trmin = min(r[5] for r in results); glob_trmax = max(r[6] for r in results)
print(f"  trust-ratio spread across modules: [{glob_trmin:.2e}, {glob_trmax:.2e}]  (LARS makes the RELATIVE step uniform ~lr regardless)")
print(f"  largest moving-and-bounded lr: {largest}   (plain SGD found NONE)")
print(f"  lrs where post-relaxation F also trends DOWN (real optimization): {learns}")
print(f"  total wall-clock {time.time()-t0:.1f}s")
print("HARD STOP (no 7b/7c/7d).")
