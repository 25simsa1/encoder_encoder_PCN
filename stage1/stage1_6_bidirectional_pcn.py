"""Stage 1.6 of PCN_FIX_PLAN.md -- the GENERATIVE / BIDIRECTIONAL direction, which NO prior stage
covered. Stages 0/1/1.5 only ran the discriminative direction (clamp image -> read label); the image
was always a CLAMPED INPUT and no edge predicts pixels from a latent, so those miniatures cannot
generate. The big model's whole purpose is BIDIRECTIONAL image<->text reconstruction, so generation
must be validated before any port. This is a small TRUE-bPC miniature (arXiv:2505.23415) with BOTH a
discriminative pathway (encoders -> shared latent -> label) AND a generative pathway (latent ->
decoders -> reconstruct each modality), one scalar energy, every update via tape.gradient. CPU only.

WHAT CARRIES OVER FROM EARLIER STAGES (so this stays honest):
 - Stage 1 fix: a clamped real-magnitude LABEL anchor (else F collapses to 0 and nothing learns).
 - Stage 1.5 fix: anchor at EVERY scale (per-scale shared latents + label heads), NOT only the top,
   else the deep credit-assignment signal vanishes. We use a per-scale anchor here too.
 - Stage 1.5 also showed conv-state relaxation is redundant at readout, so the conv tower here is a
   feed-forward autodiff edge feeding per-scale anchored latents (cheaper); the depth lesson lives in
   the per-scale anchoring, and we still check per-conv-layer gradient health (E).
 - Option-1 caveat (Rosenbaum 2021 arXiv:2106.13082): the transformer is an autodiff-through edge,
   backprop-like, not a hand-derived local PC rule. Accepted.

ARCHITECTURE.
 Modalities: IMAGE X (28x28x1) and TEXT tok (digit-class caption). Shared latents S[0..2] (one per
 image-encoder scale, all free). The IMAGE is a node that is CLAMPED when given and FREE when
 generated.
 ENCODERS (modality -> latent, discriminative):
   image: 3-conv tower (feed-forward) -> per-scale taps -> predict each S[k].
   text : embedding + 1 transformer block (softmax attention + FFN) -> per-scale text taps -> S[k].
 ANCHORS (latent -> label, discriminative, per scale -- the depth fix): y <- f(W_y_k . S[k]).
 DECODERS (latent -> modality, GENERATIVE, own weights):
   image decoder: X <- dec_img(concat S)   [dense -> sigmoid pixels]      energy ||X - dec_img(S)||^2
   text  decoder: y <- dec_txt(concat S)                                  energy ||y - dec_txt(S)||^2
 ONE energy:
   F = 0.5*mean( PI_CROSS*sum_k(||S[k]-img_tap_k||^2 [+ ||S[k]-txt_tap_k||^2 if text present])
               + PI_DISC *sum_k ||y - f(W_y_k S[k])||^2
               + PI_GEN  *( ||X - dec_img(S)||^2 + ||y - dec_txt(S)||^2 ) )
   bPC precision split PI_DISC >= PI_GEN. Every state/weight update is tape.gradient(F, .). jit. relu.

THREE DIRECTIONS TESTED (prior stages only did the first):
 1. IMAGE->TEXT (disc): clamp X, free S + y (no tokens), relax, read argmax(y).  [sanity vs Stage 1.5]
 2. TEXT->IMAGE (GENERATIVE, never tested): clamp tok + y, free X + S, relax, read the generated
    image state X. Quantified by re-classifying the generated image with direction 1 (gen-class-acc),
    plus a saved sample grid.
 3. IMAGE->IMAGE (autoencode): clamp X, free S, relax, read dec_img(S); recon error + sample grid.

PROOFS: A energy descends in BOTH clamp directions + training F down; B disc accuracy above chance;
C generation recognizable (gen-class-acc above chance + recon error + sample grids); D attention
gradient healthy in the generative (text-as-input) direction (and ~0 in image->text where text is the
OUTPUT -- reported, not hidden); E per-conv-layer gradients balanced/finite with decoders added.
PASS needs A,B,C,D,E. PARTIAL if disc works but generation is blurry/garbage (report with samples).
FAIL if energy climbs / both broken. No clips, no LARS, no d_gelu.
"""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tf.random.set_seed(0); np.random.seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- sizes ----
CONV = [(1, 8, 2), (8, 16, 2), (16, 32, 2)]          # (in,out,stride); 28->14->7->4
NUM_CONV = len(CONV)
FLAT = [14 * 14 * 8, 7 * 7 * 16, 4 * 4 * 32]
D = 64                                               # per-scale shared-latent dim
NSCALE = NUM_CONV
DMODEL = 32; HEADS = 2; HEAD_DIM = DMODEL // HEADS; L = 12; VOCAB = 12; FFN = 64
NCLASS = 10; PIX = 28 * 28

# ---- knobs ----
N_TRAIN, N_TEST, BATCH, EPOCHS = 3000, 500, 64, 30
N_INFER, BETA, ALPHA = 20, 0.1, 0.04
N_INFER_TEST = 20
EVAL_EVERY = 4
# Note: with a TRAINABLE text encoder, the bPC default (disc > gen) let the easy text/label shortcut
# dominate and the latent was not image-decodable -> generation collapsed to chance. Favoring the
# generative edges (gen > disc) forces the shared latent to stay image-decodable. This is the
# alpha_gen/alpha_disc tuning the brief sanctions, not a clip.
PI_CROSS, PI_DISC, PI_GEN = 1.0, 1.0, 2.0

# ---- data ----
(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
xtr = (xtr.astype("float32") / 255.0)[:N_TRAIN, :, :, None]
xte = (xte.astype("float32") / 255.0)[:N_TEST, :, :, None]
ytr_lab = ytr[:N_TRAIN].astype("int64"); yte_lab = yte[:N_TEST].astype("int64")
ytr_oh = tf.one_hot(ytr_lab, NCLASS).numpy().astype("float32")
def caption_table():
    cap = np.zeros((10, L), dtype=np.int32); cap[:, 0] = 1
    for d in range(10): cap[d, 1:] = 2 + d
    return cap
CAP = tf.constant(caption_table())
tok_tr = tf.gather(CAP, ytr_lab).numpy().astype("int32")

# ---- params ----
VARS, NAMES = [], []
def reg(v, n): VARS.append(v); NAMES.append(n); return v
def Wf(shape, n):
    fan_in = int(np.prod(shape[:-1]))
    return reg(tf.Variable(tf.random.normal(shape, stddev=1.0 / np.sqrt(fan_in))), n)
def Ws(shape, sd, n): return reg(tf.Variable(tf.random.normal(shape, stddev=sd)), n)
def Zz(shape, n):     return reg(tf.Variable(tf.zeros(shape)), n)

KW = [Wf([3, 3, CONV[l][0], CONV[l][1]], f"convW{l}") for l in range(NUM_CONV)]
KB = [Zz([CONV[l][1]], f"convB{l}") for l in range(NUM_CONV)]
W_i = [Wf([FLAT[k], D], f"imgtapW{k}") for k in range(NSCALE)]
b_i = [Zz([D], f"imgtapB{k}") for k in range(NSCALE)]
W_y = [Wf([D, NCLASS], f"labelW{k}") for k in range(NSCALE)]      # per-scale disc anchors
b_y = [Zz([NCLASS], f"labelB{k}") for k in range(NSCALE)]
emb = Ws([VOCAB, DMODEL], 0.10, "emb"); pos = Ws([L, DMODEL], 0.02, "pos")
Wq, Wk, Wv, Wo = Wf([DMODEL, DMODEL], "Wq"), Wf([DMODEL, DMODEL], "Wk"), Wf([DMODEL, DMODEL], "Wv"), Wf([DMODEL, DMODEL], "Wo")
Wff1, bff1 = Wf([DMODEL, FFN], "Wff1"), Zz([FFN], "bff1")
Wff2, bff2 = Wf([FFN, DMODEL], "Wff2"), Zz([DMODEL], "bff2")
W_txt = [Wf([DMODEL, D], f"txttapW{k}") for k in range(NSCALE)]   # text couples at every scale
b_txt = [Zz([D], f"txttapB{k}") for k in range(NSCALE)]
W_d0, b_d0 = Wf([NSCALE * D, 256], "decimgW0"), Zz([256], "decimgB0")   # image decoder (dense)
W_d1, b_d1 = Wf([256, PIX], "decimgW1"), Zz([PIX], "decimgB1")
W_dt, b_dt = Wf([NSCALE * D, NCLASS], "dectxtW"), Zz([NCLASS], "dectxtB")  # text decoder

IS_ATTN = [n.startswith(("emb", "pos", "Wq", "Wk", "Wv", "Wo", "Wff", "bff", "txttap")) for n in NAMES]
ATTN_VARS = [VARS[i] for i in range(len(VARS)) if IS_ATTN[i]]

def f(z): return tf.nn.relu(z)
def se(eps): return tf.reduce_sum(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)
def flat(x): return tf.reshape(x, [tf.shape(x)[0], -1])

def encode_img(X):                                    # feed-forward conv tower -> per-scale taps
    c = X; taps = []
    for l in range(NUM_CONV):
        c = f(tf.nn.conv2d(c, KW[l], [1, CONV[l][2], CONV[l][2], 1], "SAME") + KB[l])
        taps.append(f(flat(c) @ W_i[l] + b_i[l]))
    return taps

def transformer_pool(tok):
    B = tf.shape(tok)[0]
    x = tf.gather(emb, tok) + pos[None]
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    def sp(t): return tf.transpose(tf.reshape(t, [B, L, HEADS, HEAD_DIM]), [0, 2, 1, 3])
    qh, kh, vh = sp(q), sp(k), sp(v)
    att = tf.nn.softmax(tf.matmul(qh, kh, transpose_b=True) / tf.sqrt(tf.cast(HEAD_DIM, tf.float32)), axis=-1)
    ctx = tf.reshape(tf.transpose(tf.matmul(att, vh), [0, 2, 1, 3]), [B, L, DMODEL])
    x = x + ctx @ Wo
    x = x + (f(x @ Wff1 + bff1) @ Wff2 + bff2)
    return tf.reduce_mean(x, axis=1)

def encode_txt(tok):
    t = transformer_pool(tok)
    return [f(t @ W_txt[k] + b_txt[k]) for k in range(NSCALE)]

def dec_img(S):                                       # latent -> pixels (dense, sigmoid)
    h = f(tf.concat(S, axis=1) @ W_d0 + b_d0)
    o = tf.nn.sigmoid(h @ W_d1 + b_d1)
    return tf.reshape(o, [tf.shape(o)[0], 28, 28, 1])
def dec_txt(S):  return f(tf.concat(S, axis=1) @ W_dt + b_dt)
def disc(S):     return [f(S[k] @ W_y[k] + b_y[k]) for k in range(NSCALE)]

# ---- the ONE energy (per-example). text edges present only when taps_txt given ----
def energy_pe(X, taps_txt, y, S):
    taps_img = encode_img(X)
    cross = tf.zeros([tf.shape(X)[0]])
    for k in range(NSCALE):
        cross = cross + se(S[k] - taps_img[k])
        if taps_txt is not None:
            cross = cross + se(S[k] - taps_txt[k])
    dh = disc(S)
    disc_e = tf.add_n([se(y - dh[k]) for k in range(NSCALE)])
    gen = se(X - dec_img(S)) + se(y - dec_txt(S))
    return 0.5 * (PI_CROSS * cross + PI_DISC * disc_e + PI_GEN * gen)

# ---- jit steps ----
@tf.function(jit_compile=True)
def infer_train(X, tok, y, S, beta):                  # clamp X,tok,y ; free S
    taps_txt = encode_txt(tok)
    with tf.GradientTape(watch_accessed_variables=False) as t:
        t.watch(S)
        e = tf.reduce_mean(energy_pe(X, taps_txt, y, S))
    g = t.gradient(e, S)
    return [S[k] - beta * g[k] for k in range(NSCALE)], e

@tf.function(jit_compile=True)
def infer_img(X, y, S, beta):                         # clamp X ; free S,y ; NO text
    with tf.GradientTape(watch_accessed_variables=False) as t:
        t.watch(S + [y])
        e = tf.reduce_mean(energy_pe(X, None, y, S))
    g = t.gradient(e, S + [y])
    return [S[k] - beta * g[k] for k in range(NSCALE)], y - beta * g[-1], e

@tf.function(jit_compile=True)
def infer_txt2img(tok, y, X, S, beta):                # clamp tok,y ; free X,S
    taps_txt = encode_txt(tok)
    with tf.GradientTape(watch_accessed_variables=False) as t:
        t.watch([X] + S)
        e = tf.reduce_mean(energy_pe(X, taps_txt, y, S))
    g = t.gradient(e, [X] + S)
    return tf.clip_by_value(X - beta * g[0], 0.0, 1.0), [S[k] - beta * g[k + 1] for k in range(NSCALE)], e
    #         ^ pixels are values in [0,1]; clamping the IMAGE NODE to its valid range is a domain
    #           constraint on a clamped-modality readout, not an energy clip to force descent.

@tf.function(jit_compile=True)
def init_train(X, tok):
    ti = encode_img(X); tt = encode_txt(tok)
    return [0.5 * (ti[k] + tt[k]) for k in range(NSCALE)]
@tf.function(jit_compile=True)
def init_img(X):
    ti = encode_img(X); y0 = tf.add_n([f(ti[k] @ W_y[k] + b_y[k]) for k in range(NSCALE)]) / NSCALE
    return ti, y0
@tf.function(jit_compile=True)
def init_txt2img(tok):
    tt = encode_txt(tok); X0 = dec_img(tt)
    return X0, tt
@tf.function(jit_compile=True)
def learn_grads(X, tok, y, S):
    with tf.GradientTape() as t:
        taps_txt = encode_txt(tok)            # MUST be inside the tape so the text encoder
        e = tf.reduce_mean(energy_pe(X, taps_txt, y, S))   # (emb/pos/attention/FFN) gets trained
    return t.gradient(e, VARS), e

# ---- relax wrappers ----
def relax_train(X, tok, y, S, n, beta, log=False):
    bt = tf.constant(beta, tf.float32); el = []
    for _ in range(n):
        S, e = infer_train(X, tok, y, S, bt)
        if log: el.append(float(e))
    return S, el
def relax_img(X, y, S, n, beta, log=False):
    bt = tf.constant(beta, tf.float32); el = []
    for _ in range(n):
        S, y, e = infer_img(X, y, S, bt)
        if log: el.append(float(e))
    return S, y, el
def relax_txt2img(tok, y, X, S, n, beta, log=False):
    bt = tf.constant(beta, tf.float32); el = []
    for _ in range(n):
        X, S, e = infer_txt2img(tok, y, X, S, bt)
        if log: el.append(float(e))
    return X, S, el

# ---- readouts ----
def img2text(x_imgs, chunk=250):
    preds = []
    for i in range(0, len(x_imgs), chunk):
        xb = tf.constant(x_imgs[i:i + chunk]); S, y0 = init_img(xb)
        S, y, _ = relax_img(xb, y0, S, N_INFER_TEST, BETA)
        preds.append(tf.argmax(y, axis=1).numpy())
    return np.concatenate(preds)
def acc_img2text(x, lab): return float((img2text(x) == lab).mean())

def gen_from_text(classes):                           # classes: int array -> generated images [n,28,28,1]
    tok_c = tf.gather(CAP, classes); y_c = tf.one_hot(classes, NCLASS)
    X0, S = init_txt2img(tok_c)
    X, S, _ = relax_txt2img(tok_c, y_c, X0, S, N_INFER_TEST, BETA)
    return X.numpy()
def recon_image(x_imgs):                              # image->image: relax then read dec_img(S)
    xb = tf.constant(x_imgs); S, y0 = init_img(xb)
    S, y, _ = relax_img(xb, y0, S, N_INFER_TEST, BETA)
    return dec_img(S).numpy()

def near_monotone(Lst, rtol=1e-3, atol=1e-6):
    return all(Lst[i + 1] <= Lst[i] * (1 + rtol) + atol for i in range(len(Lst) - 1))

# ======================== run ========================
fb_x = tf.constant(xtr[:BATCH]); fb_tok = tf.constant(tok_tr[:BATCH]); fb_y = tf.constant(ytr_oh[:BATCH])
# A pre-train: energy descent in BOTH clamp directions
_ti0, _y0 = init_img(fb_x); _, _, elog_img_pre = relax_img(fb_x, _y0, _ti0, 30, BETA, log=True)
X0t, S0t = init_txt2img(fb_tok); _, _, elog_txt_pre = relax_txt2img(fb_tok, fb_y, X0t, S0t, 30, BETA, log=True)
print(f"[A pre ] clamp-image F {elog_img_pre[0]:.3f}->{elog_img_pre[-1]:.3f} mono={near_monotone(elog_img_pre)} | "
      f"clamp-text F {elog_txt_pre[0]:.3f}->{elog_txt_pre[-1]:.3f} mono={near_monotone(elog_txt_pre)}")

NAME2IDX = {n: i for i, n in enumerate(NAMES)}
train_energy, gn_attn, maxabs = [], [], []
gn_conv = [[] for _ in range(NUM_CONV)]
gn_decimg, gn_dectxt = [], []
acc_steps, acc_hist = [], []
step = 0; t0 = time.time()
for ep in range(EPOCHS):
    order = np.random.permutation(N_TRAIN)
    for i in range(0, N_TRAIN, BATCH):
        idx = order[i:i + BATCH]
        xb = tf.constant(xtr[idx]); tb = tf.constant(tok_tr[idx]); yb = tf.constant(ytr_oh[idx])
        S = init_train(xb, tb)
        S, _ = relax_train(xb, tb, yb, S, N_INFER, BETA)
        grads, e = learn_grads(xb, tb, yb, S)
        grads = [tf.convert_to_tensor(g) if g is not None else tf.zeros_like(v) for g, v in zip(grads, VARS)]
        for v, gv in zip(VARS, grads): v.assign_sub(ALPHA * gv)
        train_energy.append(float(e))
        for l in range(NUM_CONV): gn_conv[l].append(float(tf.norm(grads[NAME2IDX[f"convW{l}"]])))
        gn_attn.append(float(tf.linalg.global_norm([grads[j] for j in range(len(VARS)) if IS_ATTN[j]])))
        gn_decimg.append(float(tf.linalg.global_norm([grads[NAME2IDX["decimgW0"]], grads[NAME2IDX["decimgW1"]]])))
        gn_dectxt.append(float(tf.norm(grads[NAME2IDX["dectxtW"]])))
        maxabs.append(max(float(tf.reduce_max(tf.abs(v))) for v in VARS))
        step += 1
    if ep % EVAL_EVERY == 0 or ep == EPOCHS - 1:
        a = acc_img2text(xte, yte_lab); acc_steps.append(step); acc_hist.append(a)
        print(f"  epoch {ep:2d}  F={train_energy[-1]:.3f}  img->text acc={a:.3f}  "
              f"|g|conv=[{','.join(f'{gn_conv[l][-1]:.1e}' for l in range(NUM_CONV))}]  "
              f"|g|attn={gn_attn[-1]:.1e} |g|dec_img={gn_decimg[-1]:.1e} max|p|={maxabs[-1]:.2f}")
train_secs = time.time() - t0
final_acc = acc_hist[-1]

# A post-train: both directions
_, _, elog_img_post = relax_img(fb_x, init_img(fb_x)[1], init_img(fb_x)[0], 30, BETA, log=True)
X0t, S0t = init_txt2img(fb_tok); _, _, elog_txt_post = relax_txt2img(fb_tok, fb_y, X0t, S0t, 30, BETA, log=True)
print(f"[A post] clamp-image F {elog_img_post[0]:.3f}->{elog_img_post[-1]:.3f} mono={near_monotone(elog_img_post)} | "
      f"clamp-text F {elog_txt_post[0]:.3f}->{elog_txt_post[-1]:.3f} mono={near_monotone(elog_txt_post)}")

# C generation: image->image recon error, and text->image gen-class-acc (re-classify generated)
recon = recon_image(xte[:200]); recon_mse = float(np.mean((recon - xte[:200]) ** 2))
# Generation is deterministic per class, so the honest metric is per-class: generate one image for
# each digit, re-classify it with the model's own image->text path, check it comes back as that digit.
gen_one = gen_from_text(np.arange(10, dtype=np.int32))
gen_one_pred = img2text(gen_one)
per_class_ok = (gen_one_pred == np.arange(10))
gen_class_acc = float(per_class_ok.mean())
# also the 200-sample weighted version (should agree with the per-class weighted average)
gen_sample_classes = yte_lab[:200].astype(np.int32)
gen_pred200 = img2text(gen_from_text(gen_sample_classes))
gen_acc200 = float((gen_pred200 == gen_sample_classes).mean())
print(f"[C gen ] image->image recon MSE={recon_mse:.4f} (pixels in [0,1])")
print(f"[C gen ] text->image per-class (class->pred): {list(zip(range(10), gen_one_pred.tolist()))}")
print(f"[C gen ] text->image gen-class-acc(per-class)={gen_class_acc:.3f}  (200-sample weighted={gen_acc200:.3f}, chance=0.10)")

# D attention gradient in both directions (at trained model, fixed batch)
Sg = init_train(fb_x, fb_tok); Sg, _ = relax_train(fb_x, fb_tok, fb_y, Sg, N_INFER, BETA)
with tf.GradientTape() as t:                          # generative / text-as-input direction
    eg = tf.reduce_mean(energy_pe(fb_x, encode_txt(fb_tok), fb_y, Sg))
attn_gen = float(tf.linalg.global_norm([g for g in t.gradient(eg, ATTN_VARS) if g is not None] or [tf.zeros(1)]))
Si, yi0 = init_img(fb_x); Si, yi, _ = relax_img(fb_x, yi0, Si, N_INFER, BETA)
with tf.GradientTape() as t:                          # discriminative / image->text (text is OUTPUT)
    ed = tf.reduce_mean(energy_pe(fb_x, None, yi, Si))
gd = t.gradient(ed, ATTN_VARS)
attn_disc = float(tf.linalg.global_norm([g for g in gd if g is not None] or [tf.zeros(1)]))
print(f"[D attn] grad-norm generative(text-in)={attn_gen:.2e}  discriminative(text-out)={attn_disc:.2e} "
      f"(disc ~0 by design: img->text never runs the text encoder)")

# E per-conv-layer health
med_conv = np.array([np.median(gn_conv[l]) for l in range(NUM_CONV)])

# ---------------- plots ----------------
plt.figure(figsize=(6, 4))
plt.plot(elog_img_post, marker='o', ms=3, label="clamp image (relax)")
plt.plot(elog_txt_post, marker='s', ms=3, label="clamp text (generate)")
plt.xlabel("inference step"); plt.ylabel("free energy F"); plt.title("A. Inference energy descent, both directions")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "gen_energy_inference.png"), dpi=110); plt.close()

plt.figure(figsize=(6, 4)); plt.plot(train_energy, lw=0.7); plt.yscale("log")
plt.xlabel("weight update"); plt.ylabel("post-relax F"); plt.title("A. Training energy descent")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "gen_training.png"), dpi=110); plt.close()

plt.figure(figsize=(6, 4)); plt.plot(acc_steps, acc_hist, marker='o'); plt.axhline(0.1, ls='--', c='gray', label="chance")
plt.xlabel("training step"); plt.ylabel("img->text accuracy"); plt.title("B. Discriminative accuracy")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "gen_accuracy.png"), dpi=110); plt.close()

plt.figure(figsize=(8, 2.2))                          # reuse the per-class generation + its batch re-read
for c in range(10):
    plt.subplot(2, 5, c + 1); plt.imshow(gen_one[c, :, :, 0], cmap="gray", vmin=0, vmax=1)
    ok = "OK" if gen_one_pred[c] == c else "x"
    plt.title(f"text={c} -> read {gen_one_pred[c]} {ok}", fontsize=7); plt.axis("off")
plt.suptitle("C. TEXT -> IMAGE (generated, with model's own re-read)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "text2image_samples.png"), dpi=120); plt.close()

rec8 = recon_image(xte[:8])
plt.figure(figsize=(8, 2.2))
for j in range(8):
    plt.subplot(2, 8, j + 1); plt.imshow(xte[j, :, :, 0], cmap="gray", vmin=0, vmax=1); plt.axis("off")
    if j == 0: plt.ylabel("orig", fontsize=8)
    plt.subplot(2, 8, j + 9); plt.imshow(rec8[j, :, :, 0], cmap="gray", vmin=0, vmax=1); plt.axis("off")
plt.suptitle("C. IMAGE -> IMAGE (top original, bottom reconstruction)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "image2image_recon.png"), dpi=120); plt.close()

plt.figure(figsize=(7, 4))
for l in range(NUM_CONV): plt.plot(gn_conv[l], lw=0.8, label=f"convW{l}")
plt.plot(gn_attn, lw=0.8, ls='--', c='k', label="attn(all)")
plt.plot(gn_decimg, lw=0.8, ls=':', c='r', label="dec_img")
plt.xlabel("weight update"); plt.ylabel("grad norm"); plt.title("D/E. Per-layer gradient norms")
plt.yscale("log"); plt.legend(fontsize=7, ncol=2); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(HERE, "gen_grad_norms.png"), dpi=110); plt.close()

# ---------------- PASS / PARTIAL / FAIL ----------------
A_ok = (near_monotone(elog_img_pre) and near_monotone(elog_img_post) and near_monotone(elog_txt_pre)
        and near_monotone(elog_txt_post) and elog_img_post[-1] < elog_img_post[0] and elog_txt_post[-1] < elog_txt_post[0])
k = max(1, len(train_energy) // 10)
A_ok = A_ok and (np.mean(train_energy[-k:]) < np.mean(train_energy[:k]))
B_ok = final_acc > 0.30
C_ok = (gen_class_acc > 0.30) and (recon_mse < 0.08)
D_ok = (attn_gen > 0.0) and np.isfinite(attn_gen)
spread = float(med_conv.max() / (med_conv.min() + 1e-30))
E_ok = bool(np.all(np.isfinite(med_conv))) and float(med_conv.min()) > 0.0 and spread < 1e3 and float(np.max(maxabs)) < 1e3

if A_ok and B_ok and C_ok and D_ok and E_ok:
    verdict = "PASS"
elif A_ok and B_ok and (not C_ok):
    verdict = "PARTIAL"
else:
    verdict = "FAIL"

print("\n==== STAGE 1.6 (bidirectional bPC miniature, text<->image) ====")
print(f"A energy descends BOTH directions    : {A_ok}  (clamp-img {elog_img_post[0]:.2f}->{elog_img_post[-1]:.2f}, "
      f"clamp-txt {elog_txt_post[0]:.2f}->{elog_txt_post[-1]:.2f}, trainF {np.mean(train_energy[:k]):.2f}->{np.mean(train_energy[-k:]):.2f})")
print(f"B discriminative img->text works     : {B_ok}  (test acc={final_acc:.3f}, chance=0.10)")
print(f"C generation recognizable            : {C_ok}  (text->image gen-class-acc={gen_class_acc:.3f}, image->image recon MSE={recon_mse:.4f})")
print(f"D attention healthy generative dir   : {D_ok}  (gen={attn_gen:.2e}, disc={attn_disc:.2e})")
print(f"E per-conv-layer grads balanced      : {E_ok}  (median {['%.1e'%g for g in med_conv]}, spread={spread:.1e}, max|p|={np.max(maxabs):.2f})")
print(f"\nVERDICT: {verdict}")
if verdict == "PARTIAL":
    print("  PARTIAL: discriminative + energy descent work, but generation is weak (see "
          "text2image_samples.png / recon MSE). Reports what the generative direction does; informs "
          "whether the big model's bidirectional design needs richer decoders / different gen precision.")
print(f"hyperparams: D={D} d_model={DMODEL} conv={CONV} N_infer={N_INFER} beta={BETA} alpha={ALPHA} "
      f"epochs={EPOCHS} PI(cross,disc,gen)=({PI_CROSS},{PI_DISC},{PI_GEN})")
print(f"plots: gen_energy_inference / gen_training / gen_accuracy / text2image_samples / image2image_recon / gen_grad_norms")
print(f"train wall-clock: {train_secs:.1f}s (CPU, jit_compile=True)")
