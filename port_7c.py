"""7c GATE 1 -- time-boxed yes/no on text->image generation, on the validated 7.76B port.
Data: /workspace/coco7c (from prep_coco.py). Energy/LARS/architecture/forward UNCHANGED from 98d0263
(per-edge MEAN energy, LARS lr=1e-3 + bias trust floor, relax-then-step). Adds: real-data loader,
15-min checkpointing to /workspace (resume-able), divergence watch, and the 3 generation read-outs.

GENERATION PATHWAY NOTE: this Option-C port generates the image with the small decoder dec_img(S)
(64x64), NOT by inverting the deep conv encoder at 572x572. So text->image = clamp caption, relax the
shared latents S under the text side, read dec_img(S). image->image and image->text are the control
and sanity read-outs. (Full-res free-image encoder-inversion is a different mechanism this port does
not use; noted so the read-out is interpreted correctly.)

CAPS: train budget from env TRAIN_MIN (default 85 min); checkpoint every 15 min; STOP on non-finite or
F blow-up. Generation + grids after training. Then HARD STOP.
"""
import os, sys, time, gc
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.setrecursionlimit(100000)
import numpy as np
import tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from encoder_encoder_pcn import EncoderEncoderPCN, InputPCNLayer, TransposePCNLayer, FlattenPCNLayer
from dense_pcn_layer import DensePCNLayer
from conv_pcn_layer import Conv2DPCNLayer, MaxPool2DPCNLayer
from transformer_pcn_layer import AttentionPCNLayer, AddNormalizePCNLayer

print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
WORK = "/workspace/coco7c"; CKPT = os.environ.get("CKPT_DIR", "/root/ckpt7c")  # LOCAL disk (overlay, fast); the /workspace net-FS 31GB write (~11min) killed prior runs
A_CROSS, A_GEN = 1.0, 2.0
REL_C, N_INFER, LR = 0.05, 8, 1e-3
CODE = 16; REC_HW = 64; DEC_SD = 1e-3
TRAIN_S = float(os.environ.get("TRAIN_MIN", "85")) * 60
DO_CKPT = os.environ.get("DO_CKPT", "1") == "1"          # the 31GB ckpt on the network FS is ~11min/write and killed prior runs; allow disabling
CKPT_EVERY = 25 * 60; LOG_EVERY = 50; GEN_INFER = 25
t0 = time.time()

# ---- data ----
images = np.load(f"{WORK}/images.npy", mmap_mode="r")        # [N,572,572,3] float32 [0,1]
caps = np.load(f"{WORK}/captions.npy")                       # [N,192,V] one-hot
vocab = np.load(f"{WORK}/vocab.npy")                         # [V] chars
N, T, V = caps.shape
print(f"data: {N} pairs, image {images.shape}, caption {caps.shape}, vocab V={V}", flush=True)
def cap_str(onehot): return "".join(vocab[onehot.argmax(-1)]).replace("\x00", "")

# ---- model (lazy; first forward sets txt_embedding in_dim = V) ----
m = EncoderEncoderPCN(1e-4)
TXT = [l for l in m.trainable_layers if getattr(l, "share_state_layer", None) is not None]
IMG = [l.share_state_layer for l in TXT]
DIMS = [l.num_units for l in IMG]; NS = len(DIMS); betas = [REL_C * d for d in DIMS]

def fwd(layer, xi, xt, memo):
    k = id(layer)
    if k in memo: return memo[k]
    if isinstance(layer, InputPCNLayer): out = xi if layer is m.img_input else xt
    elif isinstance(layer, AddNormalizePCNLayer): out = layer(fwd(layer.prev_layers[0], xi, xt, memo), fwd(layer.prev_layers[1], xi, xt, memo))
    elif isinstance(layer, (DensePCNLayer, Conv2DPCNLayer)): out = layer(fwd(layer.prev_layer, xi, xt, memo), set_state=False)
    else: out = layer(fwd(layer.prev_layer, xi, xt, memo))
    memo[k] = out; return out
def taps(xi, xt):
    memo = {}; return [fwd(l, xi, xt, memo) for l in IMG], [fwd(l, xi, xt, memo) for l in TXT]

def Wv(shape, sd): return tf.Variable(tf.random.normal(shape, stddev=sd), trainable=True)
PROJ = [Wv([DIMS[k], CODE], DEC_SD) for k in range(NS)]
W_DI = Wv([NS * CODE, REC_HW * REC_HW * 3], DEC_SD); B_DI = tf.Variable(tf.zeros(REC_HW * REC_HW * 3))
W_DT = Wv([NS * CODE, T * V], DEC_SD); B_DT = tf.Variable(tf.zeros(T * V))
DEC_VARS = PROJ + [W_DI, B_DI, W_DT, B_DT]
def code_of(S): return tf.concat([tf.nn.relu(S[k] @ PROJ[k]) for k in range(NS)], axis=1)
def dec_img(S): return tf.nn.sigmoid(code_of(S) @ W_DI + B_DI)          # [B, 64*64*3]
def dec_txt(S): return code_of(S) @ W_DT + B_DT                         # [B, T*V]
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

X0 = tf.constant(images[0][None]); TOK0 = tf.constant(caps[0][None])
it0, tt0 = taps(X0, TOK0)
MW = [getattr(l, a) for l in m.trainable_layers for a in ("wts", "b", "gamma", "beta") if isinstance(getattr(l, a, None), tf.Variable)]
ALL_W = MW + DEC_VARS
def gidx(v): return next(i for i, w in enumerate(ALL_W) if w is v)
CONV_IDX = [gidx(v) for v in MW if len(v.shape) == 4]
HEAD_IDX = [gidx(IMG[k].wts) for k in range(NS)] + [gidx(TXT[k].wts) for k in range(NS)]
print(f"~{sum(int(np.prod(v.shape)) for v in ALL_W)/1e9:.2f}B params; {len(CONV_IDX)} conv, {NS} img+{NS} txt scales", flush=True)

def img_tgt(X): return tf.reshape(tf.image.resize(X, [REC_HW, REC_HW]), [tf.shape(X)[0], -1])
def txt_tgt(TOK): return tf.reshape(TOK, [tf.shape(TOK)[0], -1])
def F_full(S, it, tt, igt, tgt):
    cross = tf.add_n([mse(S[k] - it[k]) + mse(S[k] - tt[k]) for k in range(NS)])
    return 0.5 * tf.reduce_mean(A_CROSS * cross + A_GEN * (mse(dec_img(S) - igt) + mse(dec_txt(S) - tgt)))

@tf.function
def get_taps(xi, xt): return taps(xi, xt)
def relax_full(S, it, tt, igt, tgt, n):
    Sv = list(S)
    for _ in range(n):
        with tf.GradientTape() as tp:
            tp.watch(Sv); f = F_full(Sv, it, tt, igt, tgt)
        g = tp.gradient(f, Sv); Sv = [Sv[k] - betas[k] * g[k] for k in range(NS)]
    return Sv
@tf.function
def weight_step(xi, xt, S, igt, tgt, lr):
    with tf.GradientTape() as t:
        t.watch(ALL_W); it, tt = taps(xi, xt); F = F_full(S, it, tt, igt, tgt)
    g = t.gradient(F, ALL_W)
    for v, gg in zip(ALL_W, g):
        if gg is None: continue
        tr = (tf.norm(v) + 1e-3) / (tf.norm(gg) + 1e-6); v.assign_sub(lr * tr * gg)   # bias trust floor
    mxw = tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
    sg = tf.stack([tf.norm(g[i]) for i in (CONV_IDX + HEAD_IDX)])
    return F, mxw, sg

# ---- checkpoint / resume ----
step_v = tf.Variable(0, dtype=tf.int64)
ckpt = tf.train.Checkpoint(step=step_v, **{f"v{i}": v for i, v in enumerate(ALL_W)})
mgr = tf.train.CheckpointManager(ckpt, CKPT, max_to_keep=1)
if DO_CKPT and mgr.latest_checkpoint:
    ckpt.restore(mgr.latest_checkpoint).expect_partial(); print(f"RESUMED from {mgr.latest_checkpoint} at step {int(step_v)}", flush=True)

# ---- training (time-boxed) ----
print(f"\n[7c train] LARS lr={LR}, batch1 relax-then-step, budget {TRAIN_S/60:.0f} min, ckpt/{CKPT_EVERY/60:.0f}min", flush=True)
Fhist = []; last_ckpt = time.time(); diverged = False; step = int(step_v)
order = np.random.permutation(N)
try:
    while time.time() - t0 < TRAIN_S:
        i = int(order[step % N])
        X = tf.constant(images[i][None]); TOK = tf.constant(caps[i][None])
        igt = img_tgt(X); tgt = txt_tgt(TOK)
        it, tt = get_taps(X, TOK)
        Sv = relax_full([0.5 * (it[k] + tt[k]) for k in range(NS)], it, tt, igt, tgt, N_INFER)
        F, mxw, sg = weight_step(X, TOK, tuple(tf.constant(s) for s in Sv), igt, tgt, tf.constant(LR, tf.float32))
        F = float(F); mxw = float(mxw); mxs = max(float(tf.reduce_max(tf.abs(s))) for s in Sv)
        Fhist.append(F); step += 1; step_v.assign(step)
        if (not np.isfinite(F)) or (not np.isfinite(mxw)) or mxw > 1e3 or (len(Fhist) > 10 and F > 8 * min(Fhist) + 1):
            diverged = True
            print(f"  !! DIVERGENCE step {step}: F={F:.3e} max|w|={mxw:.3e} max|state|={mxs:.3e} -> STOP", flush=True)
            (mgr.save() if DO_CKPT else None); break
        if step % LOG_EVERY == 0:
            sgn = sg.numpy(); smin, smax = float(sgn.min()), float(sgn.max())
            print(f"  step {step:4d} t={ (time.time()-t0)/60:.1f}m  F={F:.4e} max|w|={mxw:.3e} max|s|={mxs:.3e}  gradScale[{smin:.1e},{smax:.1e}] spread={smax/(smin+1e-30):.0f}x", flush=True)
        if DO_CKPT and time.time() - last_ckpt > CKPT_EVERY:
            tc = time.time()
            try:
                mgr.save(); dt = time.time() - tc
                print(f"  [ckpt @ step {step}, {dt:.0f}s -> {CKPT}]", flush=True)
            except Exception as ce:
                dt = 999.0; print(f"  [ckpt FAILED: {ce!r}]", flush=True)
            last_ckpt = time.time()
            if dt > 120:                                  # a write must never eat the run again
                DO_CKPT = False
                print("  [ckpt write >2min/failed -> DISABLING further checkpoints to protect the run]", flush=True)
except Exception as e:
    print("TRAIN EXCEPTION:", repr(e), flush=True); (mgr.save() if DO_CKPT else None)
if DO_CKPT: mgr.save()
trained = len(Fhist)
print(f"\n[train done] steps={step} (this run {trained})  diverged={diverged}  t={(time.time()-t0)/60:.1f}m", flush=True)
if trained:
    k = max(1, trained // 10)
    print(f"  F: start {Fhist[0]:.4e} -> end {Fhist[-1]:.4e}  (mean first-{k} {np.mean(Fhist[:k]):.4e} -> last-{k} {np.mean(Fhist[-k:]):.4e})  decreasing={np.mean(Fhist[-k:])<np.mean(Fhist[:k])}", flush=True)

# ---- GENERATION READ-OUTS ----
print("\n[generation read-outs]", flush=True)
NG = min(8, N)
def relax_text(tt, tgt, n):                              # caption clamped -> drive S via text
    Sv = [tf.identity(tt[k]) for k in range(NS)]
    for _ in range(n):
        with tf.GradientTape() as tp:
            tp.watch(Sv)
            f = 0.5 * tf.reduce_mean(tf.add_n([mse(Sv[k] - tt[k]) for k in range(NS)]) + A_GEN * mse(dec_txt(Sv) - tgt))
        g = tp.gradient(f, Sv); Sv = [Sv[k] - betas[k] * g[k] for k in range(NS)]
    return Sv
def relax_image(it, igt, n):                             # image clamped -> drive S via image
    Sv = [tf.identity(it[k]) for k in range(NS)]
    for _ in range(n):
        with tf.GradientTape() as tp:
            tp.watch(Sv)
            f = 0.5 * tf.reduce_mean(tf.add_n([mse(Sv[k] - it[k]) for k in range(NS)]) + A_GEN * mse(dec_img(Sv) - igt))
        g = tp.gradient(f, Sv); Sv = [Sv[k] - betas[k] * g[k] for k in range(NS)]
    return Sv

t2i, i2i, i2i_orig, i2t_pred, i2t_true, gen_finite = [], [], [], [], [], True
for j in range(NG):
    X = tf.constant(images[j][None]); TOK = tf.constant(caps[j][None])
    it, tt = get_taps(X, TOK)
    St = relax_text(tt, txt_tgt(TOK), GEN_INFER)                       # text->image
    g = dec_img(St).numpy().reshape(REC_HW, REC_HW, 3); t2i.append(g); gen_finite &= bool(np.isfinite(g).all())
    Si = relax_image(it, img_tgt(X), GEN_INFER)                       # image->image + image->text
    i2i.append(dec_img(Si).numpy().reshape(REC_HW, REC_HW, 3)); i2i_orig.append(tf.image.resize(X, [REC_HW, REC_HW])[0].numpy())
    i2t_pred.append(cap_str(dec_txt(Si).numpy().reshape(T, V))); i2t_true.append(cap_str(caps[j]))

recon_mse = float(np.mean([(i2i[j] - i2i_orig[j]) ** 2 for j in range(NG)]))
plt.figure(figsize=(2 * NG, 4))
for j in range(NG):
    plt.subplot(2, NG, j + 1); plt.imshow(np.clip(t2i[j], 0, 1)); plt.axis("off"); plt.title(cap_str(caps[j])[:22], fontsize=6)
    plt.subplot(2, NG, NG + j + 1); plt.imshow(np.clip(i2i[j], 0, 1)); plt.axis("off")
plt.suptitle("7c  top: TEXT->IMAGE (generated)   bottom: IMAGE->IMAGE (reconstruction)", fontsize=10)
plt.tight_layout(); plt.savefig(f"{WORK}/text2image_grid.png", dpi=110); plt.close()
plt.figure(figsize=(2 * NG, 2.2))
for j in range(NG):
    plt.subplot(1, NG, j + 1); plt.imshow(np.clip(i2i_orig[j], 0, 1)); plt.axis("off"); plt.title("orig", fontsize=6)
plt.suptitle("7c image->image ORIGINALS (compare to recon row above)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{WORK}/image2image_orig.png", dpi=110); plt.close()

print(f"  text->image: generated {NG}, all finite={gen_finite}, pixel std across samples="
      f"{np.std([t.mean() for t in t2i]):.3e} (near 0 => all same = mush/mode-collapse)")
print(f"  image->image recon MSE = {recon_mse:.4f} (lower=more faithful)")
for j in range(min(4, NG)):
    print(f"  image->text[{j}] pred={repr(i2t_pred[j][:50])}  true={repr(i2t_true[j][:50])}")
print(f"  grids saved to {WORK}/text2image_grid.png , image2image_orig.png")
print(f"  total wall-clock {(time.time()-t0)/60:.1f} min")
print("HARD STOP (no Gate 2).")
