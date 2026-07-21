"""REDESIGN v2 -- reproduce the 7c collapse, then find its REAL cause.

v1 result: a tiny latent (LAT=8) does NOT cause collapse on a faithful bidirectional bench -- it
memorizes 16 pairs and text->image generates the correct, varying digits (pixel-std 0.14, recon
0.0007). So the "decode bottleneck too small" hypothesis is REFUTED at small scale; FIX1 (wider
latent) and FIX2 (anti-mean) solve a problem that does not exist here.

So what actually collapsed 7c? The diagnostic tell was max|w| moving only 1.0->1.014 over 4314 steps:
the 7.7B weights barely left initialization. A near-init decoder maps any input to ~constant output
(the data mean), and a 192-token caption mean-pooled through a near-init text encoder washes out
per-caption signal. Both are UNDERTRAINING-at-scale, not a latent bottleneck. This script holds the
bench fixed and sweeps the two things that actually differed at scale:
  (A) weight-training strength LR (does freezing weights near init reproduce the collapse?),
  (B) caption length Lcap=6 vs 192 (does mean-pooling a long caption wash out per-sample signal?),
and checks whether a wider latent or anti-mean rescues an undertrained model (they should NOT, if
undertraining is the real cause).
"""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

tf.random.set_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
N, HW, VOCAB = 16, 20, 16
DMODEL, HEADS = 32, 2; HEAD = DMODEL // HEADS
STEPS, N_INFER = 600, 10
PIX = HW * HW

(xtr, _), _ = tf.keras.datasets.mnist.load_data()
idx = np.random.permutation(len(xtr))[:N]
imgs = tf.image.resize((xtr[idx].astype("float32") / 255.0)[..., None], [HW, HW]).numpy()
DATA_STD = float(np.std(imgs[..., 0]))

def make_caps(Lcap):
    rng = np.random.RandomState(1)
    toks = rng.randint(0, VOCAB, size=(N, Lcap)).astype("int32")
    return toks, tf.one_hot(toks, VOCAB).numpy().astype("float32")

def f(z): return tf.nn.relu(z)
def mse(a, b): return tf.reduce_mean(tf.reshape(a - b, [tf.shape(a)[0], -1]) ** 2, axis=1)

def build(LAT, Lcap):
    g = tf.random.Generator.from_seed(42)
    def W(sh): return tf.Variable(g.normal(sh, stddev=1.0 / np.sqrt(np.prod(sh[:-1]))))
    def Z(sh): return tf.Variable(tf.zeros(sh))
    return dict(
        c1=W([3, 3, 1, 8]), b1=Z([8]), c2=W([3, 3, 8, 16]), b2=Z([16]),
        wi=W([5 * 5 * 16, LAT]), bi=Z([LAT]),
        emb=W([VOCAB, DMODEL]), pos=W([Lcap, DMODEL]),
        Wq=W([DMODEL, DMODEL]), Wk=W([DMODEL, DMODEL]), Wv=W([DMODEL, DMODEL]), Wo=W([DMODEL, DMODEL]),
        f1=W([DMODEL, 64]), fb1=Z([64]), f2=W([64, DMODEL]), fb2=Z([DMODEL]),
        wt=W([DMODEL, LAT]), bt=Z([LAT]),
        di0=W([LAT, 256]), dib0=Z([256]), di1=W([256, PIX]), dib1=Z([PIX]),
        dt=W([LAT, Lcap * VOCAB]), dtb=Z([Lcap * VOCAB]),
    )

def enc_img(P, x):
    h = f(tf.nn.conv2d(x, P["c1"], [1, 2, 2, 1], "SAME") + P["b1"])
    h = f(tf.nn.conv2d(h, P["c2"], [1, 2, 2, 1], "SAME") + P["b2"])
    return f(tf.reshape(h, [tf.shape(x)[0], -1]) @ P["wi"] + P["bi"])
def enc_txt(P, tk, Lcap):
    B = tf.shape(tk)[0]
    x = tf.gather(P["emb"], tk) + P["pos"][None]
    q, k, v = x @ P["Wq"], x @ P["Wk"], x @ P["Wv"]
    sp = lambda t: tf.transpose(tf.reshape(t, [B, Lcap, HEADS, HEAD]), [0, 2, 1, 3])
    a = tf.nn.softmax(tf.matmul(sp(q), sp(k), transpose_b=True) / np.sqrt(HEAD), axis=-1)
    ctx = tf.reshape(tf.transpose(tf.matmul(a, sp(v)), [0, 2, 1, 3]), [B, Lcap, DMODEL])
    x = x + ctx @ P["Wo"]; x = x + (f(x @ P["f1"] + P["fb1"]) @ P["f2"] + P["fb2"])
    return f(tf.reduce_mean(x, 1) @ P["wt"] + P["bt"])
def dec_img(P, S): return tf.nn.sigmoid(f(S @ P["di0"] + P["dib0"]) @ P["di1"] + P["dib1"])
def dec_txt(P, S): return S @ P["dt"] + P["dtb"]

def energy(P, S, il, tl, x, toh):
    cross = mse(S, il) + mse(S, tl)
    gen = mse(dec_img(P, S), tf.reshape(x, [N, -1])) + mse(dec_txt(P, S), tf.reshape(toh, [N, -1]))
    return 0.5 * tf.reduce_mean(cross + 2.0 * gen)
def infonce(il, tl, temp=0.2):
    zi = tf.math.l2_normalize(il, 1); zt = tf.math.l2_normalize(tl, 1)
    logits = zt @ tf.transpose(zi) / temp; lab = tf.range(N)
    return 0.5 * (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lab, logits)) +
                  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lab, tf.transpose(logits))))

def run(LAT, LR, anti, Lcap, label):
    toks, toh_np = make_caps(Lcap)
    P = build(LAT, Lcap); vars_ = list(P.values()); beta = 0.25 * LAT
    x = tf.constant(imgs); tk = tf.constant(toks); toh = tf.constant(toh_np)
    dec0 = [tf.identity(P[k]) for k in ("di0", "di1")]          # snapshot decoder at init (weight-movement metric)
    F0 = Fend = None
    for step in range(STEPS):
        il, tl = enc_img(P, x), enc_txt(P, tk, Lcap)
        S = 0.5 * (il + tl)
        for _ in range(N_INFER):
            with tf.GradientTape() as tp:
                tp.watch(S); e = energy(P, S, tf.constant(il), tf.constant(tl), x, toh)
            S = S - beta * tp.gradient(e, S)
        S = tf.constant(S)
        with tf.GradientTape() as tw:
            il2, tl2 = enc_img(P, x), enc_txt(P, tk, Lcap)
            loss = energy(P, S, il2, tl2, x, toh) + (anti * infonce(il2, tl2) if anti > 0 else 0.0)
        for v, gg in zip(vars_, tw.gradient(loss, vars_)):
            if gg is None: continue
            tr = (tf.norm(v) + 1e-3) / (tf.norm(gg) + 1e-6); v.assign_sub(LR * tr * gg)
        fval = float(energy(P, S, il, tl, x, toh))
        if step == 0: F0 = fval
        Fend = fval
    # weight movement of the image decoder (analog of 7c's max|w| barely moving)
    move = float((tf.norm(P["di0"] - dec0[0]) + tf.norm(P["di1"] - dec0[1])) /
                 (tf.norm(dec0[0]) + tf.norm(dec0[1]) + 1e-9))
    # text->image
    tl = enc_txt(P, tk, Lcap); S = tf.identity(tl)
    for _ in range(N_INFER * 2):
        with tf.GradientTape() as tp:
            tp.watch(S); e = 0.5 * tf.reduce_mean(mse(S, tf.constant(tl)) + 2.0 * mse(dec_txt(P, S), tf.reshape(toh, [N, -1])))
        S = S - beta * tp.gradient(e, S)
    gen = dec_img(P, S).numpy().reshape(N, HW, HW)
    # image->image recon
    il = enc_img(P, x); Si = tf.identity(il)
    for _ in range(N_INFER * 2):
        with tf.GradientTape() as tp:
            tp.watch(Si); e = 0.5 * tf.reduce_mean(mse(Si, tf.constant(il)) + 2.0 * mse(dec_img(P, Si), tf.reshape(x, [N, -1])))
        Si = Si - beta * tp.gradient(e, Si)
    rec = dec_img(P, Si).numpy().reshape(N, HW, HW)
    pix_std = float(np.mean(np.std(gen, 0))); recon = float(np.mean((rec - imgs[..., 0]) ** 2))
    collapsed = pix_std < 0.25 * DATA_STD
    print(f"  [{label}] LAT={LAT} LR={LR:.0e} anti={anti} Lcap={Lcap}: F {F0:.2e}->{Fend:.2e} | "
          f"decoder-moved={move*100:.2f}% | pixel-std={pix_std:.3e} ({pix_std/DATA_STD:.2f}x data) | "
          f"recon-MSE={recon:.4f} | COLLAPSED={collapsed}")
    return dict(label=label, gen=gen, pix_std=pix_std, recon=recon, move=move, collapsed=collapsed, Lcap=Lcap)

print(f"bench N={N}, data pixel-std={DATA_STD:.3f}. Collapse = pixel-std < {0.25*DATA_STD:.3f} (outputs nearly constant across captions).")
configs = [
    ("T-trained         ", 256,  5e-3, 0.0,   6),   # control: weights train -> should VARY
    ("U-lr1e-4          ", 256,  1e-4, 0.0,   6),   # LR sweep: weights move less
    ("U-lr3e-5          ", 256,  3e-5, 0.0,   6),
    ("U-lr1e-5 (frozen) ", 256,  1e-5, 0.0,   6),   # near-init weights -> reproduce collapse?
    ("U-wide1024 frozen ", 1024, 1e-5, 0.0,   6),   # wider latent does NOT rescue if undertraining is the cause
    ("U-tiny8 frozen    ", 8,    1e-5, 0.0,   6),
    ("U-antimean frozen ", 256,  1e-5, 1.0,   6),   # anti-mean rescue an undertrained model?
    ("L-trained Lcap192 ", 256,  5e-3, 0.0, 192),   # long caption mean-pooled: washout collapse even when trained?
]
results = []
for label, LAT, LR, anti, Lcap in configs:
    t0 = time.time(); r = run(LAT, LR, anti, Lcap, label.strip()); r["secs"] = time.time() - t0; results.append(r)

fig, axes = plt.subplots(len(results) + 1, 8, figsize=(12, 1.4 * (len(results) + 1)))
for j in range(8):
    axes[0, j].imshow(imgs[j, :, :, 0], cmap="gray", vmin=0, vmax=1); axes[0, j].axis("off")
axes[0, 0].set_title("TARGET", fontsize=7, loc="left")
for ri, r in enumerate(results):
    for j in range(8):
        axes[ri + 1, j].imshow(np.clip(r["gen"][j], 0, 1), cmap="gray", vmin=0, vmax=1); axes[ri + 1, j].axis("off")
    axes[ri + 1, 0].set_title(f"{r['label']} std={r['pix_std']:.1e} {'COLLAPSE' if r['collapsed'] else 'varies'}", fontsize=6, loc="left")
plt.suptitle("text->image: does undertraining (LR) or long captions reproduce the 7c collapse?", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "redesign_undertraining.png"), dpi=120); plt.close()

print("\n==== v2 VERDICT (cause of the 7c collapse) ====")
for r in results:
    print(f"  {r['label']:20s} decoder-moved={r['move']*100:6.2f}%  pixel-std={r['pix_std']:.3e}  recon={r['recon']:.4f}  COLLAPSED={r['collapsed']}")
print("  grid: redesign_undertraining.png")
