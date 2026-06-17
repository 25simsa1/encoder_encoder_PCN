"""PORT of the bidirectional one-energy method onto the 7.7B encoder_encoder_PCN (Option C).
PRE-7b update: (1) normalize every prediction-error term in F to a per-ELEMENT MEAN (MSE per edge,
not a raw sum over 100k-1.4M dims) to tame the 1e5-1e6 gradient runaway; (2) jit-compiled weight step
so MULTIPLE steps reuse buffers (one eager backward fit, two fragmented the pool); (3) LR sweep.
STRICT: only the energy reduction (mean vs sum) and the step execution change. No architecture / no
forward-math change. Pure forward over the existing graph, only the 5 shared latents are free, plus
the small decode-to-input anchor decided with the user.
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
REL_C, N_INFER = 0.05, 12                                # per-scale relaxation step = REL_C * dim_k (matches old per-element rate under MEAN)
IMG_SHAPE = (1, 572, 572, 3); TXT_SHAPE = (1, 192, 512)
CODE = 16; REC_HW = 64; DEC_SD = 1e-3

t0 = time.time()
m = EncoderEncoderPCN(1e-4)
TXT = [l for l in m.trainable_layers if getattr(l, "share_state_layer", None) is not None]
IMG = [l.share_state_layer for l in TXT]
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

# >>> THE NORMALIZATION FIX <<< per-ELEMENT MEAN squared error per edge (was reduce_sum over dims).
def mse(eps):  return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

def energy_parts(S, it, tt, IMG_T, TXT_T):
    cross = tf.add_n([mse(S[k] - it[k]) + mse(S[k] - tt[k]) for k in range(NS)])   # each term is a per-element MSE
    gimg = mse(dec_img(S) - IMG_T); gtxt = mse(dec_txt(S) - TXT_T)                 # decoder terms also per-element MSE
    F = 0.5 * tf.reduce_mean(A_CROSS * cross + A_GEN * (gimg + gtxt))
    return F, tf.reduce_mean(gimg), tf.reduce_mean(gtxt)

# ===================== PHASE 0: build + forward =====================
xi = tf.constant(np.random.rand(*IMG_SHAPE).astype("float32"))
xt = tf.constant(np.random.rand(*TXT_SHAPE).astype("float32") * 0.1)
print("\n[Phase 0] one feed-forward...")
it0, tt0 = taps(xi, xt)
print(f"  tap shapes ok: {[tuple(t.shape) for t in it0]}")
MW = [getattr(l, a) for l in m.trainable_layers for a in ("wts", "b", "gamma", "beta")
      if isinstance(getattr(l, a, None), tf.Variable)]
ALL_W = MW + DEC_VARS
print(f"  trainable tensors {len(ALL_W)}  (~{sum(int(np.prod(v.shape)) for v in ALL_W)/1e9:.2f}B params)")
IMG_T = tf.constant(tf.reshape(tf.image.resize(xi, [REC_HW, REC_HW]), [1, -1]))   # decode targets (constants)
TXT_T = tf.constant(tf.reshape(xt, [1, -1]))

# ===================== relaxation (per-scale beta) =====================
print("\n[relaxation] normalized F, per-scale beta = REL_C*dim_k, zero init -> must descend")
betas = [REL_C * d for d in DIMS]
itc = [tf.constant(t) for t in it0]; ttc = [tf.constant(t) for t in tt0]
Sv = [tf.Variable(tf.zeros([1, DIMS[k]])) for k in range(NS)]
Ft, GI, GT = [], [], []
for _ in range(N_INFER):
    with tf.GradientTape() as tp:
        for s in Sv: tp.watch(s)
        f, gi, gt = energy_parts(Sv, itc, ttc, IMG_T, TXT_T)
    gs = tp.gradient(f, Sv)
    for k, (s, g) in enumerate(zip(Sv, gs)): s.assign_sub(betas[k] * g)
    Ft.append(float(f)); GI.append(float(gi)); GT.append(float(gt))
F_desc = Ft[-1] < Ft[0]; mono = all(Ft[i+1] <= Ft[i]*1.0001 + 1e-9 for i in range(len(Ft)-1))
states_finite = all(bool(tf.reduce_all(tf.math.is_finite(s))) for s in Sv)
print(f"  F: {Ft[0]:.4e} -> {Ft[-1]:.4e}  descends={F_desc} monotone={mono} states_finite={states_finite}")
print(f"  CROSS-vs-DECODER balance:  image-recon MSE {GI[0]:.4e}->{GI[-1]:.4e}   text-recon MSE {GT[0]:.4e}->{GT[-1]:.4e}")
S0 = tuple(tf.constant(s) for s in Sv)
del itc, ttc, Sv, gs; gc.collect()

# ===================== jit-compiled weight step (buffer reuse -> multi-step fits) =====================
def first(pred): return next(v for v in MW if pred(v))
def gidx(v): return next(i for i, w in enumerate(ALL_W) if w is v)
GUARD = [("conv", gidx(first(lambda v: len(v.shape) == 4))),
         ("transformer", gidx(first(lambda v: len(v.shape) == 2 and int(v.shape[1]) == 1536))),
         ("img_head", gidx(IMG[0].wts)), ("txt_head", gidx(TXT[0].wts)),
         ("img_decoder", gidx(W_DI)), ("txt_decoder", gidx(W_DT))]
GIDX = [i for _, i in GUARD]

@tf.function(jit_compile=True)
def jit_step(xi, xt, S, lr):
    with tf.GradientTape() as t:
        t.watch(ALL_W)
        it, tt = taps(xi, xt)                                # FULL forward inside the tape (L2)
        cross = tf.add_n([mse(S[k] - it[k]) + mse(S[k] - tt[k]) for k in range(NS)])
        F = 0.5 * tf.reduce_mean(A_CROSS * cross + A_GEN * (mse(dec_img(S) - IMG_T) + mse(dec_txt(S) - TXT_T)))
    g = t.gradient(F, ALL_W)
    gn = tf.stack([tf.norm(g[i]) for i in GIDX])
    for w, gg in zip(ALL_W, g):
        if gg is not None:
            w.assign_sub(lr * gg)
    mx = tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
    return F, mx, gn

# ---- MEMORY PROBE + NEW per-module grad magnitudes (lr=0 -> no weight change) ----
print("\n[MEMORY PROBE + L2 grad guard] jit forward+backward at relaxed S (compiles on first call)...")
try:
    tf.config.experimental.reset_memory_stats("GPU:0")
except Exception:
    pass
t2 = time.time()
F0, mx0, gn0 = jit_step(xi, xt, S0, tf.constant(0.0))
F0 = float(F0); gn0 = gn0.numpy()
try:
    peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9
except Exception:
    peak = float("nan")
print(f"  F={F0:.4e}  compile+run {time.time()-t2:.1f}s  PEAK {peak:.1f} GB / 80 GB")
print("  NEW per-module grad magnitudes (normalized F):")
guard_ok = True
for (name, _), v in zip(GUARD, gn0):
    fin = bool(np.isfinite(v)); nz = v > 0
    print(f"    {name:14s} |grad|={v:.4e}  finite={fin} nonzero={nz}")
    guard_ok &= fin and nz
spread = float(np.max(gn0) / (np.min(gn0) + 1e-30))
print(f"  grad spread across modules = {spread:.1f}x   (was ~1e5-1e6 unnormalized)")
if not guard_ok:
    raise RuntimeError("grad guard FAILED (None/zero/nonfinite).")

# ---- multi-step confirmation + LR sweep (frozen-vs-explode) ----
print("\n[LR sweep] multi-step (jit buffer reuse); per lr ~18 steps; weights MOVE off init AND stay BOUNDED?")
cw = first(lambda v: len(v.shape) == 4); cw_init = tf.identity(cw)
results = []
for lr in [1e-4, 1e-3, 1e-2]:
    lr_t = tf.constant(lr, tf.float32); mxs = []
    for _ in range(18):
        _, mx, _ = jit_step(xi, xt, S0, lr_t); mxs.append(float(mx))
    moved = float(tf.norm(cw - cw_init) / (tf.norm(cw_init) + 1e-9))     # cumulative since init
    bounded = bool(np.isfinite(mxs[-1])) and mxs[-1] < 1e3
    results.append((lr, mxs[0], mxs[-1], moved, bounded))
    print(f"  lr={lr:.0e}: max|w| {mxs[0]:.3e} -> {mxs[-1]:.3e}   conv moved(cum)={moved:.2e}   bounded={bounded}")
bounded_lrs = [r[0] for r in results if r[3] > 1e-6 and r[4]]
largest = max(bounded_lrs) if bounded_lrs else None
multistep_ok = len(results) == 3       # got through all sweeps without OOM
print(f"  multi-step ran without OOM: {multistep_ok}")
print(f"  largest moving-and-bounded lr: {largest:.0e}" if largest else "  no lr both moved AND stayed bounded")

print("\n==== PRE-7b SUMMARY ====")
print(f"  normalization: per-element MEAN per edge (was sum over dims)")
print(f"  memory peak (jit fwd+bwd, batch1): {peak:.1f} GB / 80 GB")
print(f"  relaxation F descends: {F_desc} ({Ft[0]:.3e}->{Ft[-1]:.3e})  states finite: {states_finite}")
print(f"  NEW grad magnitudes: {[f'{v:.2e}' for v in gn0]}  spread {spread:.1f}x  (guard_ok={guard_ok})")
print(f"  decoder now contributing: image-recon {GI[0]:.3e}->{GI[-1]:.3e}, text-recon {GT[0]:.3e}->{GT[-1]:.3e}")
print(f"  LR sweep: " + "; ".join(f"{r[0]:.0e}->bounded={r[4]},moved={r[3]:.1e}" for r in results))
print(f"  largest moving-and-bounded lr = {largest}")
print(f"  multi-step (jit) fits = {multistep_ok}")
print(f"  total wall-clock {time.time()-t0:.1f}s")
print("HARD STOP (no 7b/7c/7d).")
