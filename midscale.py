"""MID-SCALE bidirectional image-text PCN -- positive control: train a shrunk version of the 7.7B
EncoderEncoderPCN TO REAL WEIGHT MOVEMENT and test generation. The 7.7B produced mush because it was
UNDERTRAINED (weights moved ~1.4% at lr=5e-4 -- too many params for the step budget). LARS makes the
per-step RELATIVE weight movement roughly scale-invariant, so at a healthy lr a mid-scale model moves
its weights tens of percent in a few hundred steps and -- per the validated bench -- should generate.

SAME architecture as the 7.7B (shrunk), SAME validated recipe (NOT the layer classes' hand-written
update rules):
  - Option-C bidirectional: conv image encoder + transformer text encoder, both predicting into shared
    latents at MULTIPLE scales, with decode-to-input pathways for BOTH modalities (image + text decoder).
  - One scalar energy F; ALL updates via tf.GradientTape.
  - LARS with the bias trust floor (+1e-3); relax-then-step; true ReLU derivative (autodiff).
  - generative precision A_GEN=2.0 >= cross precision A_CROSS=1.0 (the L4 lesson).
  - DENSE per-scale anchors: EVERY scale is anchored by BOTH an image tap and a text tap (the L3 lesson
    so depth does not freeze).

METRICS -- F is NOT a success signal (relaxation drops F while weights coast; that was the 7.7B trap):
  1. weight movement ||W-W0||/||W0|| overall + per-layer (target 40%+).
  2. generation diversity: text->image pixel-std across captions / dataset pixel-std + per-sample
     recognizability (retrieval top-1; beats-mean).
  3. sample grids: text->image and image->image (saved).
  4. F trajectory (reported only to note it dropped).
  5. image->text sanity (token accuracy vs a common-prefix baseline).

Data: MNIST, N distinct images each with a DISTINCT random caption (no class-label shortcut). MNIST
chosen over CIFAR/COCO: the goal is an unambiguous yes/no on recognizable, varying generation within a
CPU time-box; MNIST gives a clean recognizability read-out, CIFAR fidelity at this budget would be hard
to call. State of evidence, not image fidelity, is the question.
"""
import os, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

tf.random.set_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- config ----
N      = int(os.environ.get("MID_N", 64))          # distinct image-caption pairs
HW     = 28                                          # MNIST native
T, V   = 8, 32                                       # caption length, vocab (distinct random caption / image)
DM, HEADS, NBLK = 256, 4, 4; HEAD = DM // HEADS; FFN = 512
MULT   = float(os.environ.get("MID_MULT", 1.0))     # size knob
DIMS   = [int(d * MULT) for d in (4096, 4096, 2048, 2048)]   # shared-latent dim per scale (NS=4)
NS     = len(DIMS)
CODE   = 16; DEC_SD = 1e-3
A_CROSS, A_GEN = 1.0, 2.0
REL_C, N_INFER = 0.05, 8
betas  = [REL_C * d for d in DIMS]
PIX    = HW * HW
LR        = float(os.environ.get("MID_LR", 5e-3))
MAX_STEPS = int(os.environ.get("MID_STEPS", 1500))
TRAIN_MIN = float(os.environ.get("MID_TRAIN_MIN", 25))
MOVE_TARGET = 0.40
GEN_INFER = 25

# ---- data: N distinct MNIST images, each a unique random caption ----
(xtr, ytr), _ = tf.keras.datasets.mnist.load_data()
idx = np.random.permutation(len(xtr))[:N]
imgs = (xtr[idx].astype("float32") / 255.0)[..., None]                    # [N,28,28,1]
labels = ytr[idx]
DATA_STD = float(np.std(imgs[..., 0]))
MEAN_IMG = imgs[..., 0].mean(0)
_cr = np.random.RandomState(1)
toks = _cr.randint(0, V, size=(N, T)).astype("int32")
toks_oh = tf.one_hot(toks, V).numpy().astype("float32")

LEAKY = float(os.environ.get("MID_LEAKY", 0.01))   # leaky-ReLU prevents the dying-ReLU that killed the text-encoder taps in the relu run
def relu(z): return tf.nn.leaky_relu(z, alpha=LEAKY)
def mse(eps): return tf.reduce_mean(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

# ---- model params (pure-TF forward; learning via GradientTape only) ----
g = tf.random.Generator.from_seed(42)
def W(shape, sd=None):
    sd = (1.0 / np.sqrt(np.prod(shape[:-1]))) if sd is None else sd
    return tf.Variable(g.normal(shape, stddev=sd))
def Z(shape): return tf.Variable(tf.zeros(shape))

# image encoder conv stack -> features at 3 spatial scales + 1 bottleneck = NS taps
C1, C2, C3 = 32, 64, 128
f0_dim = 14 * 14 * C1; f1_dim = 7 * 7 * C2; f2_dim = 4 * 4 * C3; BN = 512
P = dict(
    c1=W([3,3,1,C1]), cb1=Z([C1]), c2=W([3,3,C1,C1]), cb2=Z([C1]),
    c3=W([3,3,C1,C2]), cb3=Z([C2]), c4=W([3,3,C2,C2]), cb4=Z([C2]),
    c5=W([3,3,C2,C3]), cb5=Z([C3]),
    wbn=W([f2_dim, BN]), bbn=Z([BN]),
    # image heads -> it[k] (dim DIMS[k]); feature dims per scale: f0,f1,f2,bottleneck
    Wi0=W([f0_dim, DIMS[0]]), bi0=Z([DIMS[0]]),
    Wi1=W([f1_dim, DIMS[1]]), bi1=Z([DIMS[1]]),
    Wi2=W([f2_dim, DIMS[2]]), bi2=Z([DIMS[2]]),
    Wi3=W([BN,     DIMS[3]]), bi3=Z([DIMS[3]]),
    # text encoder
    emb=W([V, DM]), pos=W([T, DM]),
)
for b in range(NBLK):
    P[f"Wq{b}"]=W([DM,DM]); P[f"Wk{b}"]=W([DM,DM]); P[f"Wv{b}"]=W([DM,DM]); P[f"Wo{b}"]=W([DM,DM])
    P[f"f1_{b}"]=W([DM,FFN]); P[f"fb1_{b}"]=Z([FFN]); P[f"f2_{b}"]=W([FFN,DM]); P[f"fb2_{b}"]=Z([DM])
    P[f"Wt{b}"]=W([DM, DIMS[b]]); P[f"bt{b}"]=Z([DIMS[b]])       # text head -> tt[k]
# decoders (Option-C): project each scale to CODE, concat, decode both modalities
for k in range(NS): P[f"proj{k}"]=W([DIMS[k], CODE], DEC_SD)
P["W_DI"]=W([NS*CODE, PIX], DEC_SD); P["B_DI"]=Z([PIX])
P["W_DT"]=W([NS*CODE, T*V], DEC_SD); P["B_DT"]=Z([T*V])
ALL_W = list(P.values())
P0 = {k: tf.identity(v) for k, v in P.items()}

def enc_img(x):
    h = relu(tf.nn.conv2d(x, P["c1"], 1, "SAME") + P["cb1"]); h = relu(tf.nn.conv2d(h, P["c2"], 1, "SAME") + P["cb2"])
    h = tf.nn.max_pool2d(h, 2, 2, "SAME")                                  # 14x14
    f0 = tf.reshape(h, [tf.shape(x)[0], -1])
    h = relu(tf.nn.conv2d(h, P["c3"], 1, "SAME") + P["cb3"]); h = relu(tf.nn.conv2d(h, P["c4"], 1, "SAME") + P["cb4"])
    h = tf.nn.max_pool2d(h, 2, 2, "SAME")                                  # 7x7
    f1 = tf.reshape(h, [tf.shape(x)[0], -1])
    h = relu(tf.nn.conv2d(h, P["c5"], 1, "SAME") + P["cb5"]); h = tf.nn.max_pool2d(h, 2, 2, "SAME")  # 4x4
    f2 = tf.reshape(h, [tf.shape(x)[0], -1])
    f3 = relu(f2 @ P["wbn"] + P["bbn"])
    return [relu(f0 @ P["Wi0"] + P["bi0"]), relu(f1 @ P["Wi1"] + P["bi1"]),
            relu(f2 @ P["Wi2"] + P["bi2"]), relu(f3 @ P["Wi3"] + P["bi3"])]

def enc_txt(tk):
    B = tf.shape(tk)[0]; x = tf.gather(P["emb"], tk) + P["pos"][None]; tt = []
    for b in range(NBLK):
        q, k_, v = x @ P[f"Wq{b}"], x @ P[f"Wk{b}"], x @ P[f"Wv{b}"]
        sp = lambda t: tf.transpose(tf.reshape(t, [B, T, HEADS, HEAD]), [0, 2, 1, 3])
        a = tf.nn.softmax(tf.matmul(sp(q), sp(k_), transpose_b=True) / np.sqrt(HEAD), axis=-1)
        ctx = tf.reshape(tf.transpose(tf.matmul(a, sp(v)), [0, 2, 1, 3]), [B, T, DM])
        x = x + ctx @ P[f"Wo{b}"]; x = x + (relu(x @ P[f"f1_{b}"] + P[f"fb1_{b}"]) @ P[f"f2_{b}"] + P[f"fb2_{b}"])
        pooled = tf.reduce_mean(x, 1)
        tt.append(relu(pooled @ P[f"Wt{b}"] + P[f"bt{b}"]))
    return tt

def code_of(S): return tf.concat([relu(S[k] @ P[f"proj{k}"]) for k in range(NS)], axis=1)
def dec_img(S): return tf.nn.sigmoid(code_of(S) @ P["W_DI"] + P["B_DI"])
def dec_txt(S): return code_of(S) @ P["W_DT"] + P["B_DT"]

def F_full(S, it, tt, igt, tgt):
    cross = tf.add_n([mse(S[k] - it[k]) + mse(S[k] - tt[k]) for k in range(NS)])
    return 0.5 * tf.reduce_mean(A_CROSS * cross + A_GEN * (mse(dec_img(S) - igt) + mse(dec_txt(S) - tgt)))

@tf.function
def get_taps(x, tk): return enc_img(x), enc_txt(tk)

def relax_full(S, it, tt, igt, tgt, n):
    Sv = list(S)
    for _ in range(n):
        with tf.GradientTape() as tp:
            tp.watch(Sv); f = F_full(Sv, it, tt, igt, tgt)
        gr = tp.gradient(f, Sv); Sv = [Sv[k] - betas[k] * gr[k] for k in range(NS)]
    return Sv

@tf.function
def weight_step(x, tk, S, igt, tgt, lr):
    with tf.GradientTape() as t:
        t.watch(ALL_W); it, tt = enc_img(x), enc_txt(tk); F = F_full(S, it, tt, igt, tgt)
    gr = t.gradient(F, ALL_W)
    for v, gg in zip(ALL_W, gr):
        if gg is None: continue
        tr = (tf.norm(v) + 1e-3) / (tf.norm(gg) + 1e-6); v.assign_sub(lr * tr * gg)   # LARS + bias trust floor
    mxw = tf.reduce_max(tf.stack([tf.reduce_max(tf.abs(w)) for w in ALL_W]))
    return F, mxw

def movement():
    per = {}
    for k in P:
        d0 = float(tf.norm(P0[k]))
        if d0 > 1e-6: per[k] = float(tf.norm(P[k] - P0[k])) / d0
    num = float(tf.sqrt(sum(tf.reduce_sum((P[k] - P0[k]) ** 2) for k in P)))
    den = float(tf.sqrt(sum(tf.reduce_sum(P0[k] ** 2) for k in P)))
    return num / (den + 1e-9), per

def img_tgt(x): return tf.reshape(x, [tf.shape(x)[0], -1])
def txt_tgt(tk_oh): return tf.reshape(tk_oh, [tf.shape(tk_oh)[0], -1])

NP = int(sum(int(np.prod(v.shape)) for v in ALL_W))
print(f"MID-SCALE PCN: {NP/1e6:.1f}M params | N={N} pairs | DIMS={DIMS} NS={NS} | DM={DM} blocks={NBLK}", flush=True)
print(f"recipe: A_GEN={A_GEN}>=A_CROSS={A_CROSS}, LARS+bias-floor, relax({N_INFER})-then-step, dense per-scale anchors", flush=True)
print(f"LR={LR} MAX_STEPS={MAX_STEPS} time-box={TRAIN_MIN}min  data pixel-std={DATA_STD:.3f}  chance retrieval={1/N:.3f}", flush=True)

# ---- optional LR pre-check: largest lr that MOVES weights without diverging (short) ----
if os.environ.get("MID_LRCHECK"):
    import sys
    for lr in [1e-3, 5e-3, 1e-2, 2e-2]:
        for k in P: P[k].assign(P0[k])                                   # reset
        ok = True
        for s in range(150):
            i = s % N; x = tf.constant(imgs[i][None]); tk = tf.constant(toks[i][None])
            igt = img_tgt(x); tgt = txt_tgt(tf.constant(toks_oh[i][None]))
            it, tt = get_taps(x, tk)
            Sv = relax_full([0.5*(it[k]+tt[k]) for k in range(NS)], it, tt, igt, tgt, N_INFER)
            F, mxw = weight_step(x, tk, tuple(tf.constant(s_) for s_ in Sv), igt, tgt, tf.constant(lr, tf.float32))
            if not (np.isfinite(float(F)) and float(mxw) < 1e3): ok = False; break
        mv, _ = movement()
        print(f"  LRCHECK lr={lr:.0e}: 150 steps move={mv*100:5.1f}% finite_ok={ok} F={float(F):.3e} max|w|={float(mxw):.2e}", flush=True)
    sys.exit(0)

# ---- train to real weight movement (time- and step-boxed) ----
t0 = time.time(); Fhist = []; order = np.random.permutation(N); step = 0; diverged = False
while step < MAX_STEPS and (time.time() - t0) < TRAIN_MIN * 60:
    i = int(order[step % N]); x = tf.constant(imgs[i][None]); tk = tf.constant(toks[i][None])
    igt = img_tgt(x); tgt = txt_tgt(tf.constant(toks_oh[i][None]))
    it, tt = get_taps(x, tk)
    Sv = relax_full([0.5 * (it[k] + tt[k]) for k in range(NS)], it, tt, igt, tgt, N_INFER)
    F, mxw = weight_step(x, tk, tuple(tf.constant(s_) for s_ in Sv), igt, tgt, tf.constant(LR, tf.float32))
    F = float(F); mxw = float(mxw); Fhist.append(F); step += 1
    if not (np.isfinite(F) and mxw < 1e3):
        diverged = True; print(f"  !! DIVERGENCE step {step}: F={F:.3e} max|w|={mxw:.2e} -> STOP", flush=True); break
    if step % 100 == 0:
        mv, _ = movement()
        print(f"  step {step:4d} t={(time.time()-t0)/60:.1f}m  F={F:.4e}  move={mv*100:.1f}%  max|w|={mxw:.2e}", flush=True)
        if mv >= MOVE_TARGET and step >= 300:                            # reached healthy movement -> enough
            print(f"  reached movement target ({mv*100:.1f}% >= {MOVE_TARGET*100:.0f}%) at step {step}", flush=True)
move, per = movement()
elapsed = (time.time() - t0) / 60
print(f"\n[train done] steps={step} diverged={diverged} t={elapsed:.1f}m  F {Fhist[0]:.3e}->{Fhist[-1]:.3e}  decreasing={Fhist[-1]<Fhist[0]}", flush=True)
print(f"[WEIGHT MOVEMENT] overall={move*100:.1f}%  (target {MOVE_TARGET*100:.0f}%)", flush=True)

# ---- generation read-outs ----
def relax_text(tt, tgt, n):
    Sv = [tf.identity(tt[k]) for k in range(NS)]
    for _ in range(n):
        with tf.GradientTape() as tp:
            tp.watch(Sv)
            f = 0.5 * tf.reduce_mean(tf.add_n([mse(Sv[k] - tt[k]) for k in range(NS)]) + A_GEN * mse(dec_txt(Sv) - tgt))
        gr = tp.gradient(f, Sv); Sv = [Sv[k] - betas[k] * gr[k] for k in range(NS)]
    return Sv
def relax_image(it, igt, n):
    Sv = [tf.identity(it[k]) for k in range(NS)]
    for _ in range(n):
        with tf.GradientTape() as tp:
            tp.watch(Sv)
            f = 0.5 * tf.reduce_mean(tf.add_n([mse(Sv[k] - it[k]) for k in range(NS)]) + A_GEN * mse(dec_img(Sv) - igt))
        gr = tp.gradient(f, Sv); Sv = [Sv[k] - betas[k] * gr[k] for k in range(NS)]
    return Sv

t2i = np.zeros((N, HW, HW)); i2i = np.zeros((N, HW, HW)); i2t_tokacc = []; prefix_base = []
codeT = np.zeros((N, NS * CODE)); codeI = np.zeros((N, NS * CODE))                # diagnostic: decoder input
ttvar = np.zeros((N, NS)); Stvar = np.zeros((N, NS))                              # diagnostic: per-scale norms
for j in range(N):
    x = tf.constant(imgs[j][None]); tk = tf.constant(toks[j][None])
    it, tt = get_taps(x, tk)
    St = relax_text(tt, txt_tgt(tf.constant(toks_oh[j][None])), GEN_INFER)
    t2i[j] = dec_img(St).numpy().reshape(HW, HW)
    codeT[j] = code_of(St).numpy().reshape(-1)
    ttvar[j] = [float(tf.norm(tt[k])) for k in range(NS)]; Stvar[j] = [float(tf.norm(St[k])) for k in range(NS)]
    Si = relax_image(it, img_tgt(x), GEN_INFER)
    i2i[j] = dec_img(Si).numpy().reshape(HW, HW)
    codeI[j] = code_of(Si).numpy().reshape(-1)
    pred_tok = dec_txt(Si).numpy().reshape(T, V).argmax(-1)
    i2t_tokacc.append(float(np.mean(pred_tok == toks[j])))
# DIAGNOSTIC: localize the text->image collapse (decoder is fine iff image->image varies)
print("\n---- text->image collapse diagnostic ----", flush=True)
print(f"  code(text-relaxed S): zero-frac={float(np.mean(codeT==0)):.3f}  std-across-samples={float(codeT.std(0).mean()):.3e}  mean|code|={float(np.abs(codeT).mean()):.3e}", flush=True)
print(f"  code(img -relaxed S): zero-frac={float(np.mean(codeI==0)):.3f}  std-across-samples={float(codeI.std(0).mean()):.3e}  mean|code|={float(np.abs(codeI).mean()):.3e}", flush=True)
print(f"  text taps tt: std-across-captions (per scale) = {np.round(ttvar.std(0),3).tolist()} (mean-norm {np.round(ttvar.mean(0),2).tolist()})", flush=True)
print(f"  text-relaxed S: std-across-captions (per scale) = {np.round(Stvar.std(0),3).tolist()} (mean-norm {np.round(Stvar.mean(0),2).tolist()})", flush=True)
print(f"  t2i output range across samples: max-min per pixel = {float((t2i.max(0)-t2i.min(0)).mean()):.3e} (0 => identical image for every caption)", flush=True)
# image->text baseline: predict the single most-common token everywhere (the "common prefix" trap)
mode_tok = np.bincount(toks.reshape(-1), minlength=V).argmax()
prefix_base = float(np.mean(toks == mode_tok))

# metrics
diversity = float(np.mean(np.std(t2i, 0)) / (DATA_STD + 1e-9))
d = ((t2i[:, None] - imgs[..., 0][None]) ** 2).reshape(N, N, -1).mean(-1)
retr = float(np.mean(np.argmin(d, 1) == np.arange(N)))
own = ((t2i - imgs[..., 0]) ** 2).reshape(N, -1).mean(-1)
tomean = ((t2i - MEAN_IMG[None]) ** 2).reshape(N, -1).mean(-1)
beats_mean = float(np.mean(own < tomean))
recon = float(np.mean((i2i - imgs[..., 0]) ** 2))
i2t_acc = float(np.mean(i2t_tokacc))

print(f"\n==== GENERATION METRICS (F is NOT the signal) ====", flush=True)
print(f"  text->image diversity ratio = {diversity:.3f}  (collapse~0; varies-by-input = real fraction)", flush=True)
print(f"  text->image retrieval top-1 = {retr:.3f}  (chance {1/N:.3f})", flush=True)
print(f"  text->image beats-mean      = {beats_mean:.3f}", flush=True)
print(f"  image->image recon MSE      = {recon:.4f}", flush=True)
print(f"  image->text token acc       = {i2t_acc:.3f}  vs common-token baseline {prefix_base:.3f}", flush=True)

# ---- grids ----
NGc = min(12, N)
fig, axes = plt.subplots(3, NGc, figsize=(1.1 * NGc, 3.6))
for j in range(NGc):
    axes[0, j].imshow(imgs[j, :, :, 0], cmap="gray", vmin=0, vmax=1); axes[0, j].axis("off"); axes[0, j].set_title(str(labels[j]), fontsize=7)
    axes[1, j].imshow(np.clip(t2i[j], 0, 1), cmap="gray", vmin=0, vmax=1); axes[1, j].axis("off")
    axes[2, j].imshow(np.clip(i2i[j], 0, 1), cmap="gray", vmin=0, vmax=1); axes[2, j].axis("off")
axes[0, 0].set_ylabel("target", fontsize=8); axes[1, 0].set_ylabel("text->img", fontsize=8); axes[2, 0].set_ylabel("img->img", fontsize=8)
plt.suptitle(f"Mid-scale PCN ({NP/1e6:.0f}M, move {move*100:.0f}%): top=target, mid=text->image, bottom=image->image", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "midscale_grid.png"), dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(Fhist); ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("energy F")
ax.set_title(f"F trajectory (dropped {Fhist[0]:.2e}->{Fhist[-1]:.2e}) -- reported only to note it fell, NOT a success signal")
ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "midscale_F.png"), dpi=130); plt.close()

# per-layer movement summary (weight tensors)
WMATS = [k for k in per if not k.startswith(("b", "cb", "fb")) and k not in ("bbn",)]
top = sorted(((per[k], k) for k in WMATS), reverse=True)[:8]
dump = dict(params_M=NP/1e6, N=N, DIMS=DIMS, NS=NS, DM=DM, NBLK=NBLK, LR=LR, steps=step, diverged=diverged,
            elapsed_min=elapsed, move_overall=move, move_per_layer={k: per[k] for k in WMATS},
            F0=Fhist[0], Fend=Fhist[-1], diversity=diversity, retrieval=retr, beats_mean=beats_mean,
            recon=recon, i2t_token_acc=i2t_acc, i2t_baseline=prefix_base, data_std=DATA_STD, chance=1/N)
with open(os.path.join(HERE, "midscale_results.json"), "w") as fh: json.dump(dump, fh, indent=2)

# ---- verdict ----
generates = (move >= MOVE_TARGET) and (diversity >= 0.30) and (retr > 3.0 / N)
print(f"\n==================== VERDICT ====================", flush=True)
print(f"weights moved {move*100:.1f}% (target {MOVE_TARGET*100:.0f}%); per-layer top: " +
      ", ".join(f"{k}={per[k]*100:.0f}%" for _, k in top[:5]), flush=True)
if generates:
    print(f"VERDICT: GENERATES -- with weights actually moving, text->image is VARYING and RECOGNIZABLE "
          f"(diversity={diversity:.2f}, retrieval={retr:.2f}). The 7.7B mush was UNDERTRAINING, not architecture.", flush=True)
elif move < MOVE_TARGET:
    print(f"VERDICT: INCONCLUSIVE -- weights did not reach {MOVE_TARGET*100:.0f}% movement (got {move*100:.1f}%); "
          f"need larger lr / more steps before judging generation.", flush=True)
else:
    print(f"VERDICT: STILL MUSH at {move*100:.1f}% movement (diversity={diversity:.2f}, retrieval={retr:.2f}) -- "
          f"collapse is NOT just undertraining at this scale. Reported honestly.", flush=True)
print("saved: midscale_grid.png, midscale_F.png, midscale_results.json", flush=True)
