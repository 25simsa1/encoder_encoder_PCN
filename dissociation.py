"""DISSOCIATION MATRIX -- proves generative mode-collapse in this bidirectional PCN is UNDERTRAINING
(weight-movement), not latent-capacity or the objective.

Bench (validated lineage: port_redesign.py -> port_redesign2.py): N distinct MNIST images, each paired
with a DISTINCT random caption (no shared class-label shortcut, so text->image MUST carry per-sample
content). Bidirectional model: conv image encoder + 1-block transformer text encoder -> shared latent S;
image + text decoders. Per-edge MEAN-MSE energy. LARS (trust-ratio) relax-then-step -- the validated
recipe. Fresh model per cell, identical weight init (seed 42) and data, identical everything EXCEPT the
swept axis.

>>> METRIC DISCIPLINE <<<
F (energy) DECREASING IS NOT EVIDENCE OF LEARNING -- in PC, F drops via STATE RELAXATION absorbing
per-sample error while weights coast. F is reported ONLY to show it drops in EVERY cell (so F-descent
does not track collapse). Never a success signal.

LOAD-BEARING metrics (decide every cell):
  (1) WEIGHT MOVEMENT  = ||W - W0|| / ||W0|| as %, overall AND per-layer.
  (2) DIVERSITY RATIO  = mean over pixels of std(gen across captions) / data pixel-std.
                         (collapse -> ~0; varies-by-input -> a real fraction of 1.)
  (3) RECOGNIZABILITY  = (a) retrieval top-1: for each text->image gen_i, nearest target by MSE; acc =
                             frac(argmin == i). chance = 1/N.
                         (b) beats-mean: frac of i where MSE(gen_i,target_i) < MSE(gen_i, mean image).
  (4) RECON MSE        = image->image reconstruction error (sanity).

AXES (full 2x2x2 factorial = 8 cells):
  A1 latent width : narrow (LAT_NARROW) vs wide (LAT_WIDE)
  A2 anti-mean    : off vs on (InfoNCE contrastive on the two latents)
  A3 weight move  : LOW (tiny effective LR -> weights coast, reproduce collapse) vs
                    HIGH (properly scaled LR -> weights move 50%+)
CLAIM: ONLY the weight-movement axis flips collapse. Widening the latent or adding anti-mean with LOW
movement does NOT fix collapse; HIGH movement fixes it REGARDLESS of latent width / anti-mean. If
widening the latent OR anti-mean ALSO fixes collapse, the dissociation FAILS -- reported honestly.
"""
import os, time, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

tf.random.set_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- bench sizes (credibly sized, not a toy, but CPU-affordable) ----
N, HW, VOCAB, LCAP = 32, 20, 16, 6
DMODEL, HEADS = 64, 4; HEAD = DMODEL // HEADS
FFN = 128
PIX = HW * HW
DEC_H = 512                      # image-decoder hidden width
STEPS, N_INFER = 700, 12         # may be overridden by smoke harness via env
STEPS = int(os.environ.get("DISSOC_STEPS", STEPS))
N_INFER = int(os.environ.get("DISSOC_NINFER", N_INFER))

LAT_NARROW, LAT_WIDE = 16, 512
LR_LOW, LR_HIGH = 1e-5, 8e-3     # A3: LOW coasts; HIGH chosen to move weights ~50%+ (verified, not tuned to outcome)
ANTI_STRENGTH = 1.0
TAU = 0.30                       # collapse threshold on diversity ratio (stated, applied uniformly)

# ---- data: N distinct images, each a UNIQUE random caption ----
(xtr, ytr), _ = tf.keras.datasets.mnist.load_data()
idx = np.random.permutation(len(xtr))[:N]
imgs = tf.image.resize((xtr[idx].astype("float32") / 255.0)[..., None], [HW, HW]).numpy()   # [N,HW,HW,1]
DATA_STD = float(np.std(imgs[..., 0]))
MEAN_IMG = imgs[..., 0].mean(0)                                                              # [HW,HW] dataset mean
_capr = np.random.RandomState(1)
toks = _capr.randint(0, VOCAB, size=(N, LCAP)).astype("int32")
toks_oh = tf.one_hot(toks, VOCAB).numpy().astype("float32")

def f(z): return tf.nn.relu(z)
def mse(a, b): return tf.reduce_mean(tf.reshape(a - b, [tf.shape(a)[0], -1]) ** 2, axis=1)

def build(LAT):
    g = tf.random.Generator.from_seed(42)                                                    # identical init across cells
    def W(sh): return tf.Variable(g.normal(sh, stddev=1.0 / np.sqrt(np.prod(sh[:-1]))))
    def Z(sh): return tf.Variable(tf.zeros(sh))
    return dict(
        c1=W([3, 3, 1, 16]), b1=Z([16]), c2=W([3, 3, 16, 32]), b2=Z([32]),
        wi=W([5 * 5 * 32, LAT]), bi=Z([LAT]),
        emb=W([VOCAB, DMODEL]), pos=W([LCAP, DMODEL]),
        Wq=W([DMODEL, DMODEL]), Wk=W([DMODEL, DMODEL]), Wv=W([DMODEL, DMODEL]), Wo=W([DMODEL, DMODEL]),
        f1=W([DMODEL, FFN]), fb1=Z([FFN]), f2=W([FFN, DMODEL]), fb2=Z([DMODEL]),
        wt=W([DMODEL, LAT]), bt=Z([LAT]),
        di0=W([LAT, DEC_H]), dib0=Z([DEC_H]), di1=W([DEC_H, PIX]), dib1=Z([PIX]),
        dt=W([LAT, LCAP * VOCAB]), dtb=Z([LCAP * VOCAB]),
    )

def enc_img(P, x):
    h = f(tf.nn.conv2d(x, P["c1"], [1, 2, 2, 1], "SAME") + P["b1"])
    h = f(tf.nn.conv2d(h, P["c2"], [1, 2, 2, 1], "SAME") + P["b2"])
    return f(tf.reshape(h, [tf.shape(x)[0], -1]) @ P["wi"] + P["bi"])
def enc_txt(P, tk):
    B = tf.shape(tk)[0]
    x = tf.gather(P["emb"], tk) + P["pos"][None]
    q, k, v = x @ P["Wq"], x @ P["Wk"], x @ P["Wv"]
    sp = lambda t: tf.transpose(tf.reshape(t, [B, LCAP, HEADS, HEAD]), [0, 2, 1, 3])
    a = tf.nn.softmax(tf.matmul(sp(q), sp(k), transpose_b=True) / np.sqrt(HEAD), axis=-1)
    ctx = tf.reshape(tf.transpose(tf.matmul(a, sp(v)), [0, 2, 1, 3]), [B, LCAP, DMODEL])
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

def gen_text2image(P, beta):
    tl = enc_txt(P, tf.constant(toks)); S = tf.identity(tl)
    for _ in range(N_INFER * 2):
        with tf.GradientTape() as tp:
            tp.watch(S)
            e = 0.5 * tf.reduce_mean(mse(S, tf.constant(tl)) + 2.0 * mse(dec_txt(P, S), tf.reshape(tf.constant(toks_oh), [N, -1])))
        S = S - beta * tp.gradient(e, S)
    return dec_img(P, S).numpy().reshape(N, HW, HW)

def recon_image(P, beta):
    il = enc_img(P, tf.constant(imgs)); Si = tf.identity(il)
    for _ in range(N_INFER * 2):
        with tf.GradientTape() as tp:
            tp.watch(Si)
            e = 0.5 * tf.reduce_mean(mse(Si, tf.constant(il)) + 2.0 * mse(dec_img(P, Si), tf.reshape(tf.constant(imgs), [N, -1])))
        Si = Si - beta * tp.gradient(e, Si)
    return dec_img(P, Si).numpy().reshape(N, HW, HW)

def metrics(gen, rec):
    diversity = float(np.mean(np.std(gen, 0)) / (DATA_STD + 1e-9))                            # (2)
    tgt = imgs[..., 0]                                                                        # [N,HW,HW]
    # (3a) retrieval top-1: each gen to nearest target
    d = ((gen[:, None] - tgt[None]) ** 2).reshape(N, N, -1).mean(-1)                          # [gen, target]
    retr = float(np.mean(np.argmin(d, 1) == np.arange(N)))
    # (3b) beats-mean: gen_i closer to its own target than to the mean image
    own = ((gen - tgt) ** 2).reshape(N, -1).mean(-1)
    tomean = ((gen - MEAN_IMG[None]) ** 2).reshape(N, -1).mean(-1)
    beats_mean = float(np.mean(own < tomean))
    recon = float(np.mean((rec - tgt) ** 2))                                                  # (4)
    return diversity, retr, beats_mean, recon

def weight_movement(P, P0):
    # per-layer relative movement, ONLY for nonzero-init tensors (zero-init biases have ||W0||~0 ->
    # the ratio is undefined; their movement is still captured in the overall global ratio below).
    per = {}
    for k in P:
        d0 = float(tf.norm(P0[k]))
        if d0 > 1e-6:
            per[k] = float(tf.norm(P[k] - P0[k])) / d0
    num = float(tf.sqrt(sum(tf.reduce_sum((P[k] - P0[k]) ** 2) for k in P)))                  # all params
    den = float(tf.sqrt(sum(tf.reduce_sum(P0[k] ** 2) for k in P)))
    return num / (den + 1e-9), per                                                            # overall, per-layer

def run(LAT, anti, LR, label, seed_note=""):
    P = build(LAT); P0 = {k: tf.identity(v) for k, v in P.items()}
    vars_ = list(P.values()); beta = 0.25 * LAT
    x = tf.constant(imgs); tk = tf.constant(toks); toh = tf.constant(toks_oh)
    Fhist = []
    for step in range(STEPS):
        il, tl = enc_img(P, x), enc_txt(P, tk)
        S = 0.5 * (il + tl)
        for _ in range(N_INFER):
            with tf.GradientTape() as tp:
                tp.watch(S); e = energy(P, S, tf.constant(il), tf.constant(tl), x, toh)
            S = S - beta * tp.gradient(e, S)
        S = tf.constant(S)
        with tf.GradientTape() as tw:
            il2, tl2 = enc_img(P, x), enc_txt(P, tk)
            loss = energy(P, S, il2, tl2, x, toh) + (anti * infonce(il2, tl2) if anti > 0 else 0.0)
        for v, gg in zip(vars_, tw.gradient(loss, vars_)):
            if gg is None: continue
            tr = (tf.norm(v) + 1e-3) / (tf.norm(gg) + 1e-6); v.assign_sub(LR * tr * gg)        # LARS
        Fhist.append(float(energy(P, S, il, tl, x, toh)))
    move, per = weight_movement(P, P0)
    gen = gen_text2image(P, beta); rec = recon_image(P, beta)
    diversity, retr, beats_mean, recon = metrics(gen, rec)
    collapsed = diversity < TAU
    print(f"  [{label}] LAT={LAT} anti={anti} LR={LR:.0e}: F {Fhist[0]:.2e}->{Fhist[-1]:.2e} | "
          f"move={move*100:5.1f}% | diversity={diversity:.3f} | retr={retr:.2f}(chance {1/N:.2f}) | "
          f"beats-mean={beats_mean:.2f} | recon={recon:.4f} | COLLAPSED={collapsed}")
    return dict(label=label, LAT=int(LAT), anti=float(anti), LR=float(LR), move=move, per=per,
                diversity=diversity, retr=retr, beats_mean=beats_mean, recon=recon,
                collapsed=bool(collapsed), F0=Fhist[0], Fend=Fhist[-1], Fhist=Fhist, gen=gen)

# ===================== EXPERIMENT =====================
nparams = lambda LAT: int(sum(int(np.prod(v.shape)) for v in build(LAT).values()))
print(f"bench: N={N} distinct images {imgs.shape}, captions {toks.shape} vocab={VOCAB}, DMODEL={DMODEL}")
print(f"params: narrow(LAT={LAT_NARROW})={nparams(LAT_NARROW):,}  wide(LAT={LAT_WIDE})={nparams(LAT_WIDE):,}")
print(f"data pixel-std={DATA_STD:.3f}; collapse iff diversity ratio < TAU={TAU}; chance retrieval={1/N:.3f}")
print(f"STEPS={STEPS} N_INFER={N_INFER} LR_LOW={LR_LOW:.0e} LR_HIGH={LR_HIGH:.0e}\n")

if os.environ.get("DISSOC_CAL"):
    import sys
    for _lr in [3e-3, 8e-3, 2e-2]:
        for _S in [int(s) for s in os.environ.get("DISSOC_CAL_STEPS", "400,1000,2000").split(",")]:
            STEPS = _S
            run(LAT_NARROW, 0.0, _lr, f"CAL S={_S} LR={_lr:.0e}")
    sys.exit(0)

t_all = time.time()
# ---- 2x2x2 factorial ----
print("== DISSOCIATION MATRIX (8 cells) ==")
matrix = []
for move_lvl, LR in [("LOW", LR_LOW), ("HIGH", LR_HIGH)]:
    for LAT, wname in [(LAT_NARROW, "narrow"), (LAT_WIDE, "wide")]:
        for anti, aname in [(0.0, "anti-off"), (ANTI_STRENGTH, "anti-on")]:
            lab = f"{move_lvl}/{wname}/{aname}"
            r = run(LAT, anti, LR, lab); r.update(move_lvl=move_lvl, wname=wname, aname=aname)
            matrix.append(r)

# ---- weight-movement vs collapse CURVE (fixed arch: narrow latent, anti off; sweep LR) ----
print("\n== WEIGHT-MOVEMENT vs COLLAPSE CURVE (LAT=narrow, anti off; sweep LR) ==")
curve = []
for LR in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 8e-3, 2e-2]:
    r = run(LAT_NARROW, 0.0, LR, f"curve LR={LR:.0e}"); curve.append(r)

elapsed = time.time() - t_all
print(f"\nTOTAL compute: {elapsed/60:.1f} min")

# ===================== FIGURES =====================
def cell(move_lvl, wname, aname):
    return next(r for r in matrix if r["move_lvl"] == move_lvl and r["wname"] == wname and r["aname"] == aname)

# (1) matrix heatmaps: diversity ratio, LOW vs HIGH panels (latent x antimean), annotated with retrieval
fig, axs = plt.subplots(1, 2, figsize=(9, 4))
for ax, lvl in zip(axs, ["LOW", "HIGH"]):
    M = np.array([[cell(lvl, w, a)["diversity"] for a in ("anti-off", "anti-on")] for w in ("narrow", "wide")])
    im = ax.imshow(M, vmin=0, vmax=max(0.6, M.max()), cmap="viridis")
    ax.set_xticks([0, 1], ["anti-off", "anti-on"]); ax.set_yticks([0, 1], [f"narrow({LAT_NARROW})", f"wide({LAT_WIDE})"])
    ax.set_title(f"{lvl} weight movement", fontsize=11)
    for i, w in enumerate(("narrow", "wide")):
        for j, a in enumerate(("anti-off", "anti-on")):
            c = cell(lvl, w, a)
            ax.text(j, i, f"div={c['diversity']:.2f}\nretr={c['retr']:.2f}\nmove={c['move']*100:.0f}%\n{'COLLAPSE' if c['collapsed'] else 'VARIES'}",
                    ha="center", va="center", color="white" if c["diversity"] < 0.3 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle(f"Dissociation matrix: diversity ratio (collapse iff < {TAU}). Only the LOW->HIGH axis should flip it.", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "dissoc_matrix.png"), dpi=130); plt.close()

# (2) movement-vs-collapse curve
mv = np.array([r["move"] * 100 for r in curve]); dv = np.array([r["diversity"] for r in curve])
order = np.argsort(mv); mv, dv = mv[order], dv[order]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(mv, dv, "o-", color="C0")
for r in curve: ax.annotate(f"{r['LR']:.0e}", (r["move"]*100, r["diversity"]), fontsize=6, xytext=(3,3), textcoords="offset points")
ax.axhline(TAU, ls="--", color="r", label=f"collapse threshold (div={TAU})")
above = mv[dv >= TAU]
if len(above): ax.axvline(above.min(), ls=":", color="g", label=f"breaks collapse at ~{above.min():.0f}% movement")
ax.set_xlabel("weight movement  ||W-W0||/||W0||  (%)"); ax.set_ylabel("generation diversity ratio")
ax.set_title("Weight movement vs collapse (LAT=narrow, anti off)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "dissoc_curve.png"), dpi=130); plt.close()

# (3) sample grids for key cells: collapsed-LOW vs broken-HIGH at narrow/anti-off (+ wide & anti variants under LOW)
key = [("targets", None),
       ("LOW/narrow/anti-off", cell("LOW", "narrow", "anti-off")),
       ("LOW/wide/anti-off", cell("LOW", "wide", "anti-off")),
       ("LOW/narrow/anti-on", cell("LOW", "narrow", "anti-on")),
       ("HIGH/narrow/anti-off", cell("HIGH", "narrow", "anti-off")),
       ("HIGH/wide/anti-on", cell("HIGH", "wide", "anti-on"))]
ncol = 10
fig, axes = plt.subplots(len(key), ncol, figsize=(1.1 * ncol, 1.25 * len(key)))
for ri, (name, r) in enumerate(key):
    row = imgs[:ncol, :, :, 0] if r is None else np.clip(r["gen"][:ncol], 0, 1)
    for j in range(ncol):
        axes[ri, j].imshow(row[j], cmap="gray", vmin=0, vmax=1); axes[ri, j].axis("off")
    tag = name if r is None else f"{name}  div={r['diversity']:.2f} retr={r['retr']:.2f} {'COLLAPSE' if r['collapsed'] else 'VARIES'}"
    axes[ri, 0].set_title(tag, fontsize=7, loc="left")
plt.suptitle("text->image samples (top=targets). Collapse = all columns identical (~dataset mean).", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "dissoc_samples.png"), dpi=130); plt.close()

# (4) F trajectories: low-movement (collapsed) vs high-movement -- F drops in BOTH (F is not the signal)
lo = cell("LOW", "narrow", "anti-off"); hi = cell("HIGH", "narrow", "anti-off")
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(lo["Fhist"], label=f"LOW move ({lo['move']*100:.0f}%) -> COLLAPSE (div={lo['diversity']:.2f})", color="C3")
ax.plot(hi["Fhist"], label=f"HIGH move ({hi['move']*100:.0f}%) -> VARIES (div={hi['diversity']:.2f})", color="C2")
ax.set_xlabel("training step"); ax.set_ylabel("energy F"); ax.set_yscale("log")
ax.set_title("F decreases in BOTH cells -> F-descent does NOT track collapse"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "dissoc_Ftraj.png"), dpi=130); plt.close()

# ===================== VERDICT =====================
def broke(r): return (not r["collapsed"]) and (r["retr"] > 3.0 / N)         # varies AND retrieval well above chance
low_cells = [r for r in matrix if r["move_lvl"] == "LOW"]
high_cells = [r for r in matrix if r["move_lvl"] == "HIGH"]
low_broke = [r["label"] for r in low_cells if broke(r)]
high_broke = [r["label"] for r in high_cells if broke(r)]
claim_holds = (len(low_broke) == 0) and (len(high_broke) == len(high_cells))

print("\n==================== VERDICT ====================")
print(f"LOW-movement cells that BROKE collapse (should be NONE): {low_broke or 'NONE'}")
print(f"HIGH-movement cells that BROKE collapse (should be ALL {len(high_cells)}): {len(high_broke)}/{len(high_cells)} -> {high_broke}")
if claim_holds:
    print("VERDICT: CLAIM HOLDS -- only the weight-movement axis flips collapse.")
else:
    print("VERDICT: CLAIM FAILS / PARTIAL -- see which axis also moved it (reported honestly above).")

# strip heavy arrays for json
dump = dict(meta=dict(N=N, HW=HW, DMODEL=DMODEL, LAT_NARROW=LAT_NARROW, LAT_WIDE=LAT_WIDE,
                      LR_LOW=LR_LOW, LR_HIGH=LR_HIGH, STEPS=STEPS, N_INFER=N_INFER, TAU=TAU,
                      DATA_STD=DATA_STD, chance=1.0/N, params_narrow=nparams(LAT_NARROW),
                      params_wide=nparams(LAT_WIDE), elapsed_min=elapsed/60),
            matrix=[{k: v for k, v in r.items() if k not in ("gen", "Fhist", "per")} | {"per": r["per"]} for r in matrix],
            curve=[{k: v for k, v in r.items() if k not in ("gen", "Fhist", "per")} for r in curve],
            verdict=dict(low_broke=low_broke, high_broke=high_broke, claim_holds=bool(claim_holds)))
with open(os.path.join(HERE, "dissoc_results.json"), "w") as fh: json.dump(dump, fh, indent=2)
print("\nsaved: dissoc_matrix.png, dissoc_curve.png, dissoc_samples.png, dissoc_Ftraj.png, dissoc_results.json")
