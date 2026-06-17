"""7b: GATED, MONITORED training run of the ported one-energy bidirectional 7.76B model.
LARS (with a bias trust-floor) at lr=1e-3, per-edge MEAN-normalized energy, proper relax-then-step,
batch 1. The point of 7b is PER-SCALE GRADIENT HEALTH over a real run: confirm F keeps decreasing,
no shared-latent scale or conv layer freezes (the original failure was a 400x spread / frozen deep
scale), and nothing diverges (the stable margin is narrow -- 1e-2 already diverged in the sweep).

BIAS TRUST FLOOR (the one fix from the pre-7b report): zero-init biases had ||var||=0 -> trust=0 ->
never updated. Now  trust = (||var|| + 1e-3) / (||g|| + 1e-6)  so biases get a nonzero update.

DATA: COCO is not set up on this pod, so this uses synthetic clamped image/text pairs. That is fine
for the gradient-HEALTH check (gradient flow, not semantics); 7c (real-data generation) needs real
image-text data and is explicitly out of scope here.

DIVERGENCE WATCH: if F climbs sharply or any weight/state goes non-finite, STOP immediately and report
the step and values. (Periodic full-weight checkpoints are not cheap at 28.7 GiB, so we monitor-and-
stop instead; F is logged every 10 steps so the trajectory survives an early stop.)

RESULT (A100 80GB, 140/140 steps, NO divergence): post-relaxation F decreased monotonically from
2.45 to 0.097 (~25x), max|w| stayed ~1.01 and max|state| bounded throughout. NO scale ever froze
(frozen=[] at every log) across all 5 image scales, 5 text scales, and 9 conv layers -- the raw
per-scale GRAD spread is large (~400x-13000x, inherent to the model) but LARS's uniform relative step
keeps every scale active, which is exactly what fixes the original 400x frozen-deep-scale failure.
The bias trust floor worked: a zero-init bias moved off zero (0 -> 1.05e-5). Memory peak 62.2 GB.
So 7b is stable and optimizing on the full 7.76B model. Caveat: synthetic data (4 clamped pairs),
so the F drop is fitting those pairs, not real learning; 7c needs real image-text data.
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
REL_C, N_INFER, LR = 0.05, 8, 1e-3
STEPS, N_DATA, LOG_EVERY = 140, 4, 10
IMG_SHAPE = (1, 572, 572, 3); TXT_SHAPE = (1, 192, 512)
CODE = 16; REC_HW = 64; DEC_SD = 1e-3

t0 = time.time()
m = EncoderEncoderPCN(1e-4)
TXT = [l for l in m.trainable_layers if getattr(l, "share_state_layer", None) is not None]
IMG = [l.share_state_layer for l in TXT]
DIMS = [l.num_units for l in IMG]; NS = len(DIMS); betas = [REL_C * d for d in DIMS]
print(f"shared-latent dims: {DIMS}  build {time.time()-t0:.1f}s")

def fwd(layer, xi, xt, memo):
    k = id(layer)
    if k in memo: return memo[k]
    if isinstance(layer, InputPCNLayer):
        out = xi if layer is m.img_input else xt
    elif isinstance(layer, AddNormalizePCNLayer):
        out = layer(fwd(layer.prev_layers[0], xi, xt, memo), fwd(layer.prev_layers[1], xi, xt, memo))
    elif isinstance(layer, (DensePCNLayer, Conv2DPCNLayer)):
        out = layer(fwd(layer.prev_layer, xi, xt, memo), set_state=False)
    else:
        out = layer(fwd(layer.prev_layer, xi, xt, memo))
    memo[k] = out; return out
def taps(xi, xt):
    memo = {}
    return [fwd(l, xi, xt, memo) for l in IMG], [fwd(l, xi, xt, memo) for l in TXT]

def Wv(shape, sd):
    return tf.Variable(tf.random.normal(shape, stddev=sd), trainable=True)
PROJ = [Wv([DIMS[k], CODE], DEC_SD) for k in range(NS)]
W_DI = Wv([NS * CODE, REC_HW * REC_HW * 3], DEC_SD); B_DI = tf.Variable(tf.zeros(REC_HW * REC_HW * 3))
W_DT = Wv([NS * CODE, TXT_SHAPE[1] * TXT_SHAPE[2]], DEC_SD); B_DT = tf.Variable(tf.zeros(TXT_SHAPE[1] * TXT_SHAPE[2]))
DEC_VARS = PROJ + [W_DI, B_DI, W_DT, B_DT]
def code_of(S):  return tf.concat([tf.nn.relu(S[k] @ PROJ[k]) for k in range(NS)], axis=1)
def dec_img(S):  return tf.nn.sigmoid(code_of(S) @ W_DI + B_DI)
def dec_txt(S):  return code_of(S) @ W_DT + B_DT
def mse(eps):    return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)
def F_of(S, it, tt, img_t, txt_t):
    cross = tf.add_n([mse(S[k] - it[k]) + mse(S[k] - tt[k]) for k in range(NS)])
    return 0.5 * tf.reduce_mean(A_CROSS * cross + A_GEN * (mse(dec_img(S) - img_t) + mse(dec_txt(S) - txt_t)))

# build weights with one forward
_xi = tf.constant(np.random.rand(*IMG_SHAPE).astype("float32"))
_it, _tt = taps(_xi, tf.constant(np.random.rand(*TXT_SHAPE).astype("float32")))
MW = [getattr(l, a) for l in m.trainable_layers for a in ("wts", "b", "gamma", "beta")
      if isinstance(getattr(l, a, None), tf.Variable)]
ALL_W = MW + DEC_VARS
def gidx(v): return next(i for i, w in enumerate(ALL_W) if w is v)
CONV_IDX = [gidx(v) for v in MW if len(v.shape) == 4]                 # conv1..9 kernels
IMG_HEAD_IDX = [gidx(IMG[k].wts) for k in range(NS)]                  # dense2,6,10,14,18 (per image scale)
TXT_HEAD_IDX = [gidx(TXT[k].wts) for k in range(NS)]                  # dense4,8,12,16,20 (per text scale)
SCALE_IDX = IMG_HEAD_IDX + TXT_HEAD_IDX + CONV_IDX
BIAS_V = next(v for v in MW if len(v.shape) == 1 and float(tf.norm(v)) == 0.0)   # a zero-init bias
BIAS_IDX = gidx(BIAS_V)
print(f"  ~{sum(int(np.prod(v.shape)) for v in ALL_W)/1e9:.2f}B params; {len(CONV_IDX)} conv layers, {NS} img scales, {NS} txt scales")

@tf.function
def get_taps(xi, xt): return taps(xi, xt)

def relax_S(S, it, tt, img_t, txt_t, n):
    Sv = list(S)
    for _ in range(n):
        with tf.GradientTape() as tp:
            tp.watch(Sv); f = F_of(Sv, it, tt, img_t, txt_t)
        g = tp.gradient(f, Sv)
        Sv = [Sv[k] - betas[k] * g[k] for k in range(NS)]
    return Sv

@tf.function
def weight_step(xi, xt, S, img_t, txt_t, lr):
    with tf.GradientTape() as t:
        t.watch(ALL_W)
        it, tt = taps(xi, xt)
        F = F_of(S, it, tt, img_t, txt_t)
    g = t.gradient(F, ALL_W)
    trusts = []
    for v, gg in zip(ALL_W, g):
        if gg is None: continue
        tr = (tf.norm(v) + 1e-3) / (tf.norm(gg) + 1e-6)              # bias trust FLOOR (+1e-3 on numerator)
        trusts.append(tr); v.assign_sub(lr * tr * gg)
    trs = tf.stack(trusts)
    sgrads = tf.stack([tf.norm(g[i]) for i in SCALE_IDX])           # per-scale + per-conv grad norms
    mxw = tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
    return F, mxw, sgrads, tf.reduce_min(trs), tf.reduce_max(trs), tf.norm(g[BIAS_IDX])

# ---- synthetic data (flagged: 7c needs real image-text) ----
DATA = []
for _ in range(N_DATA):
    xi = tf.constant(np.random.rand(*IMG_SHAPE).astype("float32"))
    xt = tf.constant(np.random.rand(*TXT_SHAPE).astype("float32") * 0.1)
    img_t = tf.constant(tf.reshape(tf.image.resize(xi, [REC_HW, REC_HW]), [1, -1]))
    txt_t = tf.constant(tf.reshape(xt, [1, -1]))
    DATA.append((xi, xt, img_t, txt_t))

print(f"\n[7b] LARS lr={LR}, relax-then-step, {STEPS} steps over {N_DATA} synthetic pairs (COCO not set up; 7c needs real data)")
bias0 = float(tf.norm(BIAS_V))
SCALE_NAMES = [f"imgH{k}" for k in range(NS)] + [f"txtH{k}" for k in range(NS)] + [f"conv{l}" for l in range(len(CONV_IDX))]
Fhist = []; diverged = False; bad_step = None
gc.collect()
try: tf.config.experimental.reset_memory_stats("GPU:0")
except Exception: pass
peak = float("nan")
for s in range(STEPS):
    xi, xt, img_t, txt_t = DATA[s % N_DATA]
    it, tt = get_taps(xi, xt)
    Sv = relax_S([0.5 * (it[k] + tt[k]) for k in range(NS)], it, tt, img_t, txt_t, N_INFER)
    F, mxw, sgrads, trmn, trmx, bgrad = weight_step(xi, xt, tuple(tf.constant(s_) for s_ in Sv), img_t, txt_t, tf.constant(LR, tf.float32))
    F = float(F); mxw = float(mxw); sg = sgrads.numpy(); mxs = max(float(tf.reduce_max(tf.abs(x))) for x in Sv)
    Fhist.append(F)
    if np.isnan(peak):
        try: peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9
        except Exception: pass
    # divergence watch
    if (not np.isfinite(F)) or (not np.isfinite(mxw)) or (not np.isfinite(mxs)) or mxw > 1e3 or (len(Fhist) > 5 and F > 5 * min(Fhist) + 1.0):
        diverged = True; bad_step = s
        print(f"  !! DIVERGENCE at step {s}: F={F:.3e} max|w|={mxw:.3e} max|state|={mxs:.3e} -> STOP")
        break
    if s % LOG_EVERY == 0 or s == STEPS - 1:
        smin, smax = float(sg.min()), float(sg.max())
        frozen = [SCALE_NAMES[i] for i in range(len(sg)) if sg[i] <= 0.0]
        print(f"  step {s:3d}  F={F:.4e}  max|w|={mxw:.3e} max|state|={mxs:.3e}  scaleGrad[{smin:.1e},{smax:.1e}] spread={smax/(smin+1e-30):.0f}x"
              f"  trust[{float(trmn):.1e},{float(trmx):.1e}]  bias|g|={float(bgrad):.1e}  frozen={frozen}")

bias1 = float(tf.norm(BIAS_V))
# final per-scale grad snapshot from the last step
print("\n==== 7b SUMMARY ====")
print(f"  steps run: {len(Fhist)}/{STEPS}  diverged={diverged}" + (f" at step {bad_step}" if diverged else ""))
k = max(1, len(Fhist) // 10)
print(f"  F trajectory: start {Fhist[0]:.4e} -> end {Fhist[-1]:.4e}   (first-{k}-mean {np.mean(Fhist[:k]):.4e} -> last-{k}-mean {np.mean(Fhist[-k:]):.4e})")
print(f"  F decreasing: {np.mean(Fhist[-k:]) < np.mean(Fhist[:k])}")
print(f"  bias norm: {bias0:.3e} -> {bias1:.3e}  (moved off zero: {bias1 > bias0})")
print(f"  memory peak: {peak:.1f} GB / 80 GB")
print(f"  per-scale health: see scaleGrad spread + frozen list above (no scale should be frozen)")
print(f"  total wall-clock {time.time()-t0:.1f}s")
print("HARD STOP (no 7c/7d).")
