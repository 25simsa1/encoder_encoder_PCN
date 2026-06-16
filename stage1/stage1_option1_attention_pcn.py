"""Stage 1, OPTION 1 of PCN_FIX_PLAN.md  -- small MULTIMODAL predictive-coding model with a REAL
softmax-attention block in the text branch, trained from ONE scalar free energy F, every update via
tf.GradientTape (autodiff through every block, including the attention). This is the method proven in
stage0/stage0_mnist_pcn_jit.py, scaled up to a faithful MINIATURE of encoder_encoder_PCN's shape:
a conv image branch and a transformer text branch that BOTH predict into one shared latent (the
bidirectional / shared-latent coupling, cf. bPC arXiv:2505.23415).

We do NOT reuse the big repo's hand-written update_state / update_wts / update_b -- those inconsistent
hand-derivatives (d_gelu used for a relu forward, ad-hoc averaging, LARS rescaling) are exactly the
broken thing this plan replaces. There is ONE energy F and EVERY gradient is tape.gradient(F, .), so
the activation derivative f' is consistent by construction.

ARCHITECTURE (single shared free latent). The branch "latent" is each encoder's OUTPUT (its
prediction of the shared latent); the SHARED LATENT s is the one free PC state both branches predict
and from which the label is decoded:
   image  --conv_tower-->  predicts s
   caption--text_tower-->  predicts s          (text_tower contains the softmax-attention block)
   s      --W_y-->         predicts the clamped one-hot label y
This short chain (encoder edge + label edge) is the structure that made Stage 0 learn. An earlier
version put 3-5 free latents between image and label; the relaxation then absorbed the label mismatch
by moving states and the conv weights got a negligible gradient (accuracy stuck at chance, and two
different image-path depths produced byte-identical output -- the tell that the image path was inert).
Giving each encoder a DIRECT target (the relaxed s) fixes credit assignment.

OPTION-1 framing & caveat (do not forget): "autodiff through every block" -- the conv tower and the
transformer are differentiable edges; their params (incl. attention Q/K/V/O/FFN) get full autodiff
gradients. During INFERENCE the encoder outputs are CONSTANT w.r.t. the free latent s (inputs clamped,
params fixed), so inference energy descent is unaffected by the attention nonlinearity; during
LEARNING the attention params receive their gradient by autodiff back through softmax. Per Rosenbaum
2021 (arXiv:2106.13082), PC-as-autodiff-through-an-arbitrary-graph relies on the fixed-prediction
assumption and is backprop-LIKE, NOT strictly a free-energy-following local rule. So the attention
edge is trained correctly-as-backprop, not by a hand-derived local PC rule. That is the accepted
Option-1 trade.

WHY A LABEL-ANCHOR EDGE. With both modalities as clamped *sources* meeting at a *free* latent, F has a
trivial collapse minimum (everything -> 0 => F -> 0). Standard discriminative PC -- and Stage 0 --
avoids this by clamping the label as a real-magnitude TARGET. The shared latent must predict the
clamped one-hot y; this anchors F and supplies the learning signal.

THE ENERGY (defined once; everything is a gradient of THIS).
 Clamped: x_img, tok (caption tokens), y (one-hot label).   Free: s (shared latent).
 Edges, each  eps = target - mu,  f = relu (true relu derivative is automatic under autodiff):
   1. s <- conv_tower(x_img)      (image encoder edge)
   2. s <- text_tower(tok)        (text encoder edge, softmax attention inside)
   3. y <- W_y . s                (label anchor; y clamped one-hot)
 F = 0.5 * mean_over_batch( PI_RECON*(||s-conv||^2 + ||s-text||^2) + PI_LABEL*||y - f(W_y s)||^2 ).
 Precision weights (bPC, alpha_disc > alpha_gen): the label edge is up-weighted so its signal drives
 s and, via the recon edges, both encoders.

TWO PHASES, both tape.gradient on the SAME F, both @tf.function(jit_compile=True):
 INFERENCE: relax s by GD on F for N_INFER steps (clamped inputs untouched), beta~0.1.
 LEARNING : one SGD step on ALL params (conv tower, label head, AND the attention Q/K/V/O/FFN) via
            tape.gradient(F, params), alpha.  No LARS, no Adam, no clips -- we want to SEE F.

READOUT (cross-modal, image -> label): PC-faithful energy classification. Clamp the image, and for
each candidate digit c in 0..9 clamp the text branch to caption(c) AND the anchor to one-hot(c),
relax s, pick the c with the lowest per-example energy. The true class is the one where the image
encoder agrees with the (caption, label)-driven shared latent.

CPU only. Proof, not a benchmark.
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

# ---- sizes (small miniature; not the big model's sizes) ----
D       = 64
DMODEL  = 64
HEADS   = 4
HEAD_DIM = DMODEL // HEADS
L       = 16
VOCAB   = 12          # 0 PAD, 1 CLS, 2..11 = digit 0..9
FFN     = 128
CFLAT   = 7 * 7 * 16
NCLASS  = 10

# ---- training knobs ----
N_TRAIN, N_TEST, BATCH, EPOCHS = 5000, 500, 64, 45
N_INFER, BETA, ALPHA = 25, 0.1, 0.05
N_INFER_TEST = 25
PI_RECON, PI_LABEL = 1.0, 5.0          # bPC-style precision; label (discriminative) edge up-weighted

# ---- data: MNIST image  <->  digit-class token sequence  +  one-hot label anchor ----
(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
xtr = (xtr.astype("float32") / 255.0)[:N_TRAIN, :, :, None]
xte = (xte.astype("float32") / 255.0)[:N_TEST, :, :, None]
ytr_lab = ytr[:N_TRAIN].astype("int64"); yte_lab = yte[:N_TEST].astype("int64")
ytr_oh = tf.one_hot(ytr_lab, NCLASS).numpy().astype("float32")

def caption_table():
    cap = np.zeros((10, L), dtype=np.int32)
    cap[:, 0] = 1                       # CLS
    for d in range(10):
        cap[d, 1:] = 2 + d              # digit token (distinct per class)
    return cap
CAP = tf.constant(caption_table())                        # [10, L]
tok_tr = tf.gather(CAP, ytr_lab).numpy().astype("int32")  # [N_TRAIN, L]

# ---- params (plain tf.Variables; updated ONLY by autodiff SGD in the learn step) ----
def W(shape):
    fan_in = int(np.prod(shape[:-1]))
    return tf.Variable(tf.random.normal(shape, stddev=1.0 / np.sqrt(fan_in)))   # fan-in (LeCun) init
def Wsd(shape, sd): return tf.Variable(tf.random.normal(shape, stddev=sd))
def Z(shape):       return tf.Variable(tf.zeros(shape))
conv1, b1 = W([3, 3, 1, 8]), Z([8])
conv2, b2 = W([3, 3, 8, 16]), Z([16])
W_img, b_img = W([CFLAT, D]), Z([D])                       # conv tower head -> D
W_y,  b_y  = W([D, NCLASS]), Z([NCLASS])                   # label-anchor head
emb = Wsd([VOCAB, DMODEL], 0.10)
pos = Wsd([L, DMODEL], 0.02)
Wq, Wk, Wv, Wo = W([DMODEL, DMODEL]), W([DMODEL, DMODEL]), W([DMODEL, DMODEL]), W([DMODEL, DMODEL])
Wff1, bff1 = W([DMODEL, FFN]), Z([FFN])
Wff2, bff2 = W([FFN, DMODEL]), Z([DMODEL])
W_txt, b_txt = W([DMODEL, D]), Z([D])                      # text tower head -> D

OTHER = [conv1, b1, conv2, b2, W_img, b_img, W_y, b_y]                                # conv/dense/label
ATTN  = [emb, pos, Wq, Wk, Wv, Wo, Wff1, bff1, Wff2, bff2, W_txt, b_txt]              # text branch incl. attention
PARAMS = OTHER + ATTN
N_OTHER = len(OTHER)

def f(z): return tf.nn.relu(z)

# ---- the two ENCODER edges (each a full differentiable block predicting s) ----
def conv_tower(x_img):
    h = f(tf.nn.conv2d(x_img, conv1, [1, 2, 2, 1], "SAME") + b1)    # 28 -> 14
    h = f(tf.nn.conv2d(h, conv2, [1, 2, 2, 1], "SAME") + b2)        # 14 -> 7
    h = tf.reshape(h, [tf.shape(h)[0], CFLAT])
    return f(h @ W_img + b_img)                                     # [B,D]

def text_tower(tok):
    B = tf.shape(tok)[0]
    x = tf.gather(emb, tok) + pos[None]                            # [B,L,DMODEL]
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    def split(t):
        return tf.transpose(tf.reshape(t, [B, L, HEADS, HEAD_DIM]), [0, 2, 1, 3])
    qh, kh, vh = split(q), split(k), split(v)
    scores = tf.matmul(qh, kh, transpose_b=True) / tf.sqrt(tf.cast(HEAD_DIM, tf.float32))
    att = tf.nn.softmax(scores, axis=-1)                          # <-- the softmax attention
    ctx = tf.matmul(att, vh)
    ctx = tf.reshape(tf.transpose(ctx, [0, 2, 1, 3]), [B, L, DMODEL])
    x = x + ctx @ Wo                                              # residual after attention
    x = x + (f(x @ Wff1 + bff1) @ Wff2 + bff2)                    # residual after FFN
    return f(tf.reduce_mean(x, axis=1) @ W_txt + b_txt)           # mean-pool -> [B,D]

def mu_lab(s): return f(s @ W_y + b_y)                            # label-anchor prediction

def se(eps):
    return tf.reduce_sum(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)

# ---- the ONE energy, two views (reduced uses precomputed CONSTANT encoder outputs c_img, c_txt) ----
def reduced_per_ex(s, c_img, c_txt, y):
    recon = se(s - c_img) + se(s - c_txt)
    label = se(y - mu_lab(s))
    return 0.5 * (PI_RECON * recon + PI_LABEL * label)

def full_per_ex(x_img, tok, s, y):
    recon = se(s - conv_tower(x_img)) + se(s - text_tower(tok))
    label = se(y - mu_lab(s))
    return 0.5 * (PI_RECON * recon + PI_LABEL * label)

# ---- jit-compiled steps (both phases are tape.gradient of the SAME F) ----
@tf.function(jit_compile=True)
def encoders(x_img, tok):
    return conv_tower(x_img), text_tower(tok)

@tf.function(jit_compile=True)
def infer_step(s, c_img, c_txt, y, beta):
    with tf.GradientTape(watch_accessed_variables=False) as t:
        t.watch(s)
        e = tf.reduce_mean(reduced_per_ex(s, c_img, c_txt, y))
    gs = t.gradient(e, s)
    return s - beta * gs, e

@tf.function(jit_compile=True)
def reduced_scalar(s, c_img, c_txt, y):
    return tf.reduce_mean(reduced_per_ex(s, c_img, c_txt, y))

@tf.function(jit_compile=True)
def energy_per_ex_at(s, c_img, c_txt, y):
    return reduced_per_ex(s, c_img, c_txt, y)

@tf.function(jit_compile=True)
def learn_grads(x_img, tok, y, s):
    with tf.GradientTape() as t:           # watches all trainable PARAMS automatically
        e = tf.reduce_mean(full_per_ex(x_img, tok, s, y))
    return t.gradient(e, PARAMS), e         # full F => conv + attention params get autodiff grads

# ---- relax helper ----
def relax(s, c_img, c_txt, y, n, beta, log=False):
    bt = tf.constant(beta, tf.float32); elog = []
    for _ in range(n):
        s, e = infer_step(s, c_img, c_txt, y, bt)
        if log: elog.append(float(e))
    if log: elog.append(float(reduced_scalar(s, c_img, c_txt, y)))
    return s, elog

def init_s(c_img, c_txt):
    return 0.5 * (c_img + c_txt)           # feed-forward init (average of the two encoder votes)

# ---- cross-modal readout: argmin-energy over the 10 candidate (caption, one-hot) pairs ----
def classify(x_imgs, chunk=250):
    preds = []
    for i in range(0, len(x_imgs), chunk):
        xb = tf.constant(x_imgs[i:i + chunk]); B = xb.shape[0]
        c_img, _ = encoders(xb, tf.tile(CAP[0][None], [B, 1]))     # image encoder same for all candidates
        energies = []
        for c in range(NCLASS):
            tok_c = tf.tile(CAP[c][None], [B, 1])
            y_c = tf.tile(tf.one_hot(c, NCLASS)[None], [B, 1])
            _, c_txt = encoders(xb, tok_c)
            s, _ = relax(init_s(c_img, c_txt), c_img, c_txt, y_c, N_INFER_TEST, BETA)
            energies.append(energy_per_ex_at(s, c_img, c_txt, y_c).numpy())   # [B]
        preds.append(np.argmin(np.stack(energies, axis=1), axis=1))           # [B]
    return np.concatenate(preds)

def accuracy(x_imgs, labels):
    return float((classify(x_imgs) == labels).mean())

def near_monotone(Lst, rtol=1e-3, atol=1e-6):
    return all(Lst[i + 1] <= Lst[i] * (1 + rtol) + atol for i in range(len(Lst) - 1))

# ======================== run ========================
fb_x = tf.constant(xtr[:BATCH]); fb_tok = tf.constant(tok_tr[:BATCH]); fb_y = tf.constant(ytr_oh[:BATCH])
ci0, ct0 = encoders(fb_x, fb_tok)
_, infer_pre = relax(init_s(ci0, ct0), ci0, ct0, fb_y, 30, BETA, log=True)
print(f"[A pre-train ] F {infer_pre[0]:.3f} -> {infer_pre[-1]:.3f}  monotone={near_monotone(infer_pre)}")

train_energy, gn_attn, gn_other, maxabs = [], [], [], []
acc_steps, acc_tr, acc_te = [], [], []
step = 0; t0 = time.time()
for ep in range(EPOCHS):
    order = np.random.permutation(N_TRAIN)
    for i in range(0, N_TRAIN, BATCH):
        idx = order[i:i + BATCH]
        xb = tf.constant(xtr[idx]); tb = tf.constant(tok_tr[idx]); yb = tf.constant(ytr_oh[idx])
        c_img, c_txt = encoders(xb, tb)
        s, _ = relax(init_s(c_img, c_txt), c_img, c_txt, yb, N_INFER, BETA)   # phase 1: relax s
        grads, e = learn_grads(xb, tb, yb, s)                                 # phase 2: dF/dparams at relaxed s
        grads = [tf.convert_to_tensor(g) for g in grads]                      # densify embedding IndexedSlices
        for v, g in zip(PARAMS, grads):
            v.assign_sub(ALPHA * g)                                           # raw SGD, no clip/LARS/Adam
        train_energy.append(float(e))
        gn_other.append(float(tf.linalg.global_norm(grads[:N_OTHER])))
        gn_attn.append(float(tf.linalg.global_norm(grads[N_OTHER:])))
        maxabs.append(max(float(tf.reduce_max(tf.abs(v))) for v in PARAMS))
        step += 1
    if ep % 2 == 0 or ep == EPOCHS - 1:
        at = accuracy(xtr[:400], ytr_lab[:400]); ae = accuracy(xte, yte_lab)
        acc_steps.append(step); acc_tr.append(at); acc_te.append(ae)
        print(f"  epoch {ep:2d}  F={train_energy[-1]:.3f}  train_acc={at:.3f}  test_acc={ae:.3f}"
              f"  |g|attn={gn_attn[-1]:.2e} |g|other={gn_other[-1]:.2e} max|p|={maxabs[-1]:.2f}")
train_secs = time.time() - t0
final_test, final_train = acc_te[-1], acc_tr[-1]

cip, ctp = encoders(fb_x, fb_tok)
_, infer_post = relax(init_s(cip, ctp), cip, ctp, fb_y, 30, BETA, log=True)
print(f"[A post-train] F {infer_post[0]:.3f} -> {infer_post[-1]:.3f}  monotone={near_monotone(infer_post)}")

# ---------------- plots ----------------
plt.figure(figsize=(6, 4))
plt.plot(infer_pre, marker='o', ms=3, label="untrained")
plt.plot(infer_post, marker='s', ms=3, label="trained")
plt.xlabel("inference step"); plt.ylabel("free energy F"); plt.title("A. Inference energy descent")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(HERE, "energy_inference.png"), dpi=110); plt.close()

plt.figure(figsize=(6, 4)); plt.plot(train_energy, lw=0.7)
plt.xlabel("weight update"); plt.ylabel("post-relaxation F"); plt.title("B. Training energy descent")
plt.yscale("log"); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(HERE, "energy_training.png"), dpi=110); plt.close()

plt.figure(figsize=(6, 4))
plt.plot(acc_steps, acc_tr, marker='o', label="train"); plt.plot(acc_steps, acc_te, marker='s', label="test")
plt.axhline(0.1, ls='--', c='gray', label="chance"); plt.xlabel("training step"); plt.ylabel("cross-modal accuracy")
plt.title("C. image -> label (argmin-energy readout)"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(HERE, "accuracy.png"), dpi=110); plt.close()

plt.figure(figsize=(6, 4))
plt.plot(gn_attn, lw=0.8, label="attention params"); plt.plot(gn_other, lw=0.8, label="conv/dense/label params")
plt.xlabel("weight update"); plt.ylabel("grad global-norm"); plt.title("D. Gradient norms (attention vs rest)")
plt.yscale("log"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(HERE, "grad_norms.png"), dpi=110); plt.close()

# ---------------- PASS / FAIL ----------------
A_ok = near_monotone(infer_pre) and near_monotone(infer_post) and (infer_post[-1] < infer_post[0])
k = max(1, len(train_energy) // 10)
B_ok = np.mean(train_energy[-k:]) < np.mean(train_energy[:k])
C_ok = final_test > 0.50          # honest "clearly learning" bar (chance=0.10)
attn_finite = bool(np.all(np.isfinite(gn_attn))) and bool(np.all(np.isfinite(maxabs)))
attn_nonzero = float(np.min(gn_attn)) > 0.0
bounded = float(np.max(maxabs)) < 1e3
D_ok = attn_finite and attn_nonzero and bounded
PASS = A_ok and B_ok and C_ok and D_ok

print("\n==== STAGE 1 (Option 1: autodiff-through-attention, single energy) ====")
print(f"A inference energy down (monotone)   : {A_ok}  (post-train {infer_post[0]:.3f} -> {infer_post[-1]:.3f})")
print(f"B training energy trends down        : {B_ok}  ({np.mean(train_energy[:k]):.3f} -> {np.mean(train_energy[-k:]):.3f})")
print(f"C cross-modal acc clearly learning   : {C_ok}  (test={final_test:.3f}, train={final_train:.3f}, chance=0.10, bar=0.50)")
print(f"D attention grads finite/nonzero/bdd : {D_ok}  "
      f"(|g|attn range {np.min(gn_attn):.2e}..{np.max(gn_attn):.2e}, "
      f"|g|other range {np.min(gn_other):.2e}..{np.max(gn_other):.2e}, "
      f"attn/other ratio~{np.median(np.array(gn_attn) / np.array(gn_other)):.2e}, max|p|={np.max(maxabs):.2f})")
print(f"\nVERDICT: {'PASS' if PASS else 'FAIL'}")
print(f"hyperparams: D={D} d_model={DMODEL} heads={HEADS} L={L}  N_infer={N_INFER} beta={BETA} alpha={ALPHA} "
      f"epochs={EPOCHS} batch={BATCH} N_train={N_TRAIN}  PI(recon,label)=({PI_RECON},{PI_LABEL})  (no clips, raw SGD)")
print(f"train wall-clock: {train_secs:.1f}s (CPU, jit_compile=True)")
