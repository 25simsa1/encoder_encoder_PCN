"""REDESIGN experiment (miniature, CPU) -- does widening the latent and/or an anti-mean objective
break the mode-collapse-to-mean that 7c showed on the big model?

Bench: a small bidirectional model in the SAME regime that collapsed in 7c -- distinct images paired
with distinct captions (no easy shared class label), per-edge MEAN-MSE energy, LARS relax-then-step.
Each caption maps to ONE specific image, so SUCCESS = text->image produces the per-caption image and
the outputs VARY across captions (high pixel-std); COLLAPSE = every caption decodes to the dataset
mean (near-zero pixel-std), exactly the 7c failure. We first reproduce the collapse (tiny latent),
then ablate the fixes. The PASS signal is VARIATION-by-input + recognizability, NOT "F decreased"
(mean-collapse also lowers F -- that is the trap).

Configs: C0 baseline (tiny latent, plain mean-MSE) -> expect collapse. C1 wide latent. C2 tiny +
anti-mean (InfoNCE contrastive on the latents). C3 wide + anti-mean.
"""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

tf.random.set_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
N, HW, L, VOCAB = 16, 20, 6, 16          # 16 distinct images, 20x20, captions = 6 tokens over vocab 16
DMODEL, HEADS = 32, 2; HEAD = DMODEL // HEADS
STEPS, N_INFER, LR = 600, 10, 5e-3

# ---- data: N distinct MNIST images + a unique random caption per image ----
(xtr, ytr), _ = tf.keras.datasets.mnist.load_data()
idx = np.random.permutation(len(xtr))[:N]
imgs = tf.image.resize((xtr[idx].astype("float32") / 255.0)[..., None], [HW, HW]).numpy()   # [N,HW,HW,1]
toks = np.random.randint(0, VOCAB, size=(N, L)).astype("int32")                              # unique caption per image
toks_oh = tf.one_hot(toks, VOCAB).numpy().astype("float32")                                  # [N,L,VOCAB]
PIX = HW * HW

def f(z): return tf.nn.relu(z)
def mse(a, b): return tf.reduce_mean(tf.reshape(a - b, [tf.shape(a)[0], -1]) ** 2, axis=1)    # per-element MEAN per edge (the 7c objective)

def build(LAT):
    g = tf.random.Generator.from_seed(42)
    def W(sh): return tf.Variable(g.normal(sh, stddev=1.0 / np.sqrt(np.prod(sh[:-1]))))
    def Z(sh): return tf.Variable(tf.zeros(sh))
    P = dict(
        c1=W([3, 3, 1, 8]), b1=Z([8]), c2=W([3, 3, 8, 16]), b2=Z([16]),
        wi=W([5 * 5 * 16, LAT]), bi=Z([LAT]),                       # image encoder -> LAT
        emb=W([VOCAB, DMODEL]), pos=W([L, DMODEL]),
        Wq=W([DMODEL, DMODEL]), Wk=W([DMODEL, DMODEL]), Wv=W([DMODEL, DMODEL]), Wo=W([DMODEL, DMODEL]),
        f1=W([DMODEL, 64]), fb1=Z([64]), f2=W([64, DMODEL]), fb2=Z([DMODEL]),
        wt=W([DMODEL, LAT]), bt=Z([LAT]),                           # text encoder -> LAT
        di0=W([LAT, 256]), dib0=Z([256]), di1=W([256, PIX]), dib1=Z([PIX]),   # image decoder (LAT is the bottleneck)
        dt=W([LAT, L * VOCAB]), dtb=Z([L * VOCAB]),                 # text decoder
    )
    return P

def enc_img(P, x):
    h = f(tf.nn.conv2d(x, P["c1"], [1, 2, 2, 1], "SAME") + P["b1"])     # 20->10
    h = f(tf.nn.conv2d(h, P["c2"], [1, 2, 2, 1], "SAME") + P["b2"])     # 10->5
    return f(tf.reshape(h, [tf.shape(x)[0], -1]) @ P["wi"] + P["bi"])
def enc_txt(P, tk):
    B = tf.shape(tk)[0]
    x = tf.gather(P["emb"], tk) + P["pos"][None]
    q, k, v = x @ P["Wq"], x @ P["Wk"], x @ P["Wv"]
    sp = lambda t: tf.transpose(tf.reshape(t, [B, L, HEADS, HEAD]), [0, 2, 1, 3])
    a = tf.nn.softmax(tf.matmul(sp(q), sp(k), transpose_b=True) / np.sqrt(HEAD), axis=-1)
    ctx = tf.reshape(tf.transpose(tf.matmul(a, sp(v)), [0, 2, 1, 3]), [B, L, DMODEL])
    x = x + ctx @ P["Wo"]; x = x + (f(x @ P["f1"] + P["fb1"]) @ P["f2"] + P["fb2"])
    return f(tf.reduce_mean(x, 1) @ P["wt"] + P["bt"])
def dec_img(P, S): return tf.nn.sigmoid(f(S @ P["di0"] + P["dib0"]) @ P["di1"] + P["dib1"])     # [B,PIX]
def dec_txt(P, S): return S @ P["dt"] + P["dtb"]                                                 # [B,L*VOCAB]

def energy(P, S, il, tl, x, toh):
    cross = mse(S, il) + mse(S, tl)
    gen = mse(dec_img(P, S), tf.reshape(x, [N, -1])) + mse(dec_txt(P, S), tf.reshape(toh, [N, -1]))
    return 0.5 * tf.reduce_mean(cross + 2.0 * gen)
def infonce(il, tl, temp=0.2):                                       # anti-mean: latents must distinguish samples
    zi = tf.math.l2_normalize(il, 1); zt = tf.math.l2_normalize(tl, 1)
    logits = zt @ tf.transpose(zi) / temp
    lab = tf.range(N)
    return 0.5 * (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lab, logits)) +
                  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lab, tf.transpose(logits))))

def run(LAT, anti, label):
    P = build(LAT); vars_ = list(P.values()); beta = 0.25 * LAT
    x = tf.constant(imgs); tk = tf.constant(toks); toh = tf.constant(toks_oh)
    Fhist = []
    for step in range(STEPS):
        il, tl = enc_img(P, x), enc_txt(P, tk)                       # encoders (constants for the S relaxation)
        S = 0.5 * (il + tl)
        for _ in range(N_INFER):                                     # relax S on the mean-MSE energy
            with tf.GradientTape() as tp:
                tp.watch(S); e = energy(P, S, tf.constant(il), tf.constant(tl), x, toh)
            S = S - beta * tp.gradient(e, S)
        S = tf.constant(S)
        with tf.GradientTape() as tw:                                # weight step (LARS), recompute encoders inside
            il2, tl2 = enc_img(P, x), enc_txt(P, tk)
            loss = energy(P, S, il2, tl2, x, toh) + (anti * infonce(il2, tl2) if anti > 0 else 0.0)
        gs = tw.gradient(loss, vars_)
        for v, gg in zip(vars_, gs):
            if gg is None: continue
            tr = (tf.norm(v) + 1e-3) / (tf.norm(gg) + 1e-6)
            v.assign_sub(LR * tr * gg)
        Fhist.append(float(energy(P, S, il, tl, x, toh)))
    # ---- text->image generation: clamp caption, relax S (text side), decode ----
    tl = enc_txt(P, tk); S = tf.identity(tl)
    for _ in range(N_INFER * 2):
        with tf.GradientTape() as tp:
            tp.watch(S); e = 0.5 * tf.reduce_mean(mse(S, tf.constant(tl)) + 2.0 * mse(dec_txt(P, S), tf.reshape(toh, [N, -1])))
        S = S - beta * tp.gradient(e, S)
    gen = dec_img(P, S).numpy().reshape(N, HW, HW)                   # text->image
    # image->image recon
    il = enc_img(P, x); Si = tf.identity(il)
    for _ in range(N_INFER * 2):
        with tf.GradientTape() as tp:
            tp.watch(Si); e = 0.5 * tf.reduce_mean(mse(Si, tf.constant(il)) + 2.0 * mse(dec_img(P, Si), tf.reshape(x, [N, -1])))
        Si = Si - beta * tp.gradient(e, Si)
    rec = dec_img(P, Si).numpy().reshape(N, HW, HW)
    pix_std = float(np.mean(np.std(gen, axis=0)))                    # std ACROSS the N generated images, per-pixel, averaged
    recon_mse = float(np.mean((rec - imgs[..., 0]) ** 2))
    # gen variation vs the dataset mean (how far each gen is from the mean gen)
    var_ratio = float(np.mean(np.std(gen, 0)) / (np.std(imgs[..., 0]) + 1e-9))
    print(f"  [{label}] LAT={LAT} anti={anti}: F {Fhist[0]:.3e}->{Fhist[-1]:.3e} | pixel-std-across-samples={pix_std:.4e} | recon-MSE={recon_mse:.4f} | gen/data std-ratio={var_ratio:.3f}")
    return dict(label=label, LAT=LAT, anti=anti, gen=gen, rec=rec, pix_std=pix_std, recon_mse=recon_mse, var_ratio=var_ratio, F=Fhist)

print(f"bench: N={N} distinct images {imgs.shape}, captions {toks.shape} (vocab {VOCAB}), per-edge MEAN-MSE + LARS")
print("dataset image pixel-std (for scale):", float(np.std(imgs[..., 0])))
configs = [("C0 baseline tiny", 8, 0.0), ("C1 wide-256", 256, 0.0), ("C1 wide-1024", 1024, 0.0),
           ("C2 tiny+antimean", 8, 1.0), ("C3 wide+antimean", 256, 1.0)]
results = []
for label, LAT, anti in configs:
    t0 = time.time(); r = run(LAT, anti, label); r["secs"] = time.time() - t0; results.append(r)

# ---- grids: for each config, top=target images, bottom=text->image generations ----
fig, axes = plt.subplots(len(results) + 1, 8, figsize=(12, 1.4 * (len(results) + 1)))
for j in range(8):
    axes[0, j].imshow(imgs[j, :, :, 0], cmap="gray", vmin=0, vmax=1); axes[0, j].axis("off")
axes[0, 0].set_ylabel("TARGET", fontsize=7)
for ri, r in enumerate(results):
    for j in range(8):
        axes[ri + 1, j].imshow(np.clip(r["gen"][j], 0, 1), cmap="gray", vmin=0, vmax=1); axes[ri + 1, j].axis("off")
    axes[ri + 1, 0].set_title(f"{r['label']} std={r['pix_std']:.1e}", fontsize=6, loc="left")
plt.suptitle("text->image by config (top row = targets). Collapse = all columns identical.", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "redesign_text2image.png"), dpi=120); plt.close()

base = results[0]["pix_std"]
print("\n==== REDESIGN VERDICT ====")
print(f"  baseline C0 pixel-std={base:.3e}  (collapse reproduced if this is near 0 vs data-std {np.std(imgs[...,0]):.3f})")
for r in results:
    factor = r["pix_std"] / (base + 1e-12)
    varies = r["pix_std"] > 0.3 * np.std(imgs[..., 0])     # outputs vary if std is a real fraction of data std
    print(f"  {r['label']:18s} LAT={r['LAT']:5d} anti={r['anti']}: pixel-std={r['pix_std']:.3e} ({factor:.1f}x baseline)  recon-MSE={r['recon_mse']:.4f}  VARIES_BY_INPUT={varies}")
winners = [r["label"] for r in results if r["pix_std"] > 0.3 * np.std(imgs[..., 0])]
print(f"\n  configs that VARY by input (broke collapse): {winners if winners else 'NONE'}")
print("  grid: redesign_text2image.png")
