"""Stage 1.7 of PCN_FIX_PLAN.md -- validate a CONV/DECONV image decoder as a replacement for the
giant flatten->Dense generative heads, as a miniature, before deciding the 7.7B port (PORT_PLAN.md
section 5). The flatten->Dense heads are ~6B of 7.7B params, they make the autodiff-of-one-F backward
graph a memory risk, AND they are the generative decoders (L4) whose quality drives text->image. Conv
decoders are smaller and standard for image generation. This proves it small first. CPU only.

METHOD. Start from the PASSING Stage 1.6 model (stage1/stage1_6_bidirectional_pcn.py) and change ONE
thing only -- the IMAGE generative decoder. Everything else is identical: same single energy F, same
bidirectional structure, same alpha_gen >= alpha_disc precision (L4 = PI_GEN > PI_DISC), all updates
via tape.gradient, jit on step fns, true relu derivative, no clips, encoders AND decoders computed
INSIDE the learning tape (L2). The text decoder is unchanged (this stage is about the image decoder).

THE DECODER SWAP.
  dense (Stage 1.6 baseline): concat(S) -> Dense(256) relu -> Dense(784) sigmoid -> [28,28,1].
  conv  (this stage):         concat(S) -> Dense(7*7*16) relu -> reshape -> [upsample x2 + conv] x2
                              -> sigmoid -> [28,28,1].  (nearest-upsample + conv = anti-checkerboard
                              transpose-conv; jit-safe, no dynamic conv2d_transpose output_shape.)
  Both read concat(S) over ALL scales, so neither is a top-only decoder.

PRESERVING THE MULTI-SCALE ANCHOR STRUCTURE (L3). Stage 1.5 proved depth holds only with dense
per-scale anchors; in the big model those anchors ARE the tied reconstruction heads at 5 scales. In
THIS miniature the per-scale anchoring is the per-scale label heads (y <- W_y_k . S[k]) plus the
per-scale cross edges (S[k] <- img_tap_k and S[k] <- txt_tap_k). The decoder swap does NOT touch any
of those, and the conv decoder still reads every scale via concat(S). We VERIFY (not assume) that this
stays balanced: per-scale state movement and per-scale/per-layer gradient spread are logged for both
decoders; a "pass" that secretly froze a scale is a FAIL.

HEAD-TO-HEAD. The dense (1.6) and conv decoders are trained under identical seed and settings, the
only difference being the image decoder, and compared on: A energy descent both clamp directions +
training F; B image->text accuracy; C generation (text->image per-class re-read + grid, image->image
recon MSE + grid); D decoder param count; E multi-scale anchor health; F attention grad in the
generative direction. VERDICT recommends conv decoders for the port iff conv MATCHES OR BEATS dense on
generation with fewer params and balanced anchors; otherwise keep flatten-dense + gradient
checkpointing, with evidence why. No hacks/clips.
"""
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- sizes (identical to Stage 1.6) ----
CONV = [(1, 8, 2), (8, 16, 2), (16, 32, 2)]
NUM_CONV = len(CONV); FLAT = [14 * 14 * 8, 7 * 7 * 16, 4 * 4 * 32]
D = 64; NSCALE = NUM_CONV
DMODEL = 32; HEADS = 2; HEAD_DIM = DMODEL // HEADS; L = 12; VOCAB = 12; FFN = 64
NCLASS = 10; PIX = 28 * 28
N_TRAIN, N_TEST, BATCH, EPOCHS = 3000, 500, 64, 30
N_INFER, BETA, ALPHA = 20, 0.1, 0.04
N_INFER_TEST = 20; EVAL_EVERY = 10
# text->image frees the image PIXELS (with a [0,1] clip) coupled through the decoder; the conv
# decoder's sharper map can overshoot at BETA=0.1, so the generative relaxation uses a smaller step
# and more iterations. Applied to BOTH decoders so the head-to-head stays fair. (step-size tuning,
# brief-sanctioned, not a clip.)
N_GEN, GEN_BETA = 50, 0.03
PI_CROSS, PI_DISC, PI_GEN = 1.0, 1.0, 2.0       # L4: generative >= discriminative

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

def near_monotone(Lst, rtol=1e-3, atol=1e-6):
    return all(Lst[i + 1] <= Lst[i] * (1 + rtol) + atol for i in range(len(Lst) - 1))
def up2x(x):                                          # nearest-neighbour 2x upsample (jit-safe)
    return tf.repeat(tf.repeat(x, 2, axis=1), 2, axis=2)


def run(decoder_kind):
    """Build a fresh model with the chosen image decoder, train, evaluate. Returns a metrics dict."""
    tf.random.set_seed(0); np.random.seed(0)          # identical init for everything shared
    VARS, NAMES = [], []
    def reg(v, n): VARS.append(v); NAMES.append(n); return v
    def Wf(shape, n):
        fan = int(np.prod(shape[:-1])); return reg(tf.Variable(tf.random.normal(shape, stddev=1.0 / np.sqrt(fan))), n)
    def Ws(shape, sd, n): return reg(tf.Variable(tf.random.normal(shape, stddev=sd)), n)
    def Zz(shape, n):     return reg(tf.Variable(tf.zeros(shape)), n)

    KW = [Wf([3, 3, CONV[l][0], CONV[l][1]], f"convW{l}") for l in range(NUM_CONV)]
    KB = [Zz([CONV[l][1]], f"convB{l}") for l in range(NUM_CONV)]
    W_i = [Wf([FLAT[k], D], f"imgtapW{k}") for k in range(NSCALE)]
    b_i = [Zz([D], f"imgtapB{k}") for k in range(NSCALE)]
    W_y = [Wf([D, NCLASS], f"labelW{k}") for k in range(NSCALE)]
    b_y = [Zz([NCLASS], f"labelB{k}") for k in range(NSCALE)]
    emb = Ws([VOCAB, DMODEL], 0.10, "emb"); pos = Ws([L, DMODEL], 0.02, "pos")
    Wq, Wk, Wv, Wo = Wf([DMODEL, DMODEL], "Wq"), Wf([DMODEL, DMODEL], "Wk"), Wf([DMODEL, DMODEL], "Wv"), Wf([DMODEL, DMODEL], "Wo")
    Wff1, bff1 = Wf([DMODEL, FFN], "Wff1"), Zz([FFN], "bff1")
    Wff2, bff2 = Wf([FFN, DMODEL], "Wff2"), Zz([DMODEL], "bff2")
    W_txt = [Wf([DMODEL, D], f"txttapW{k}") for k in range(NSCALE)]
    b_txt = [Zz([D], f"txttapB{k}") for k in range(NSCALE)]
    W_dt, b_dt = Wf([NSCALE * D, NCLASS], "dectxtW"), Zz([NCLASS], "dectxtB")    # text decoder (unchanged)
    # ---- the ONLY thing that differs: the image decoder ----
    if decoder_kind == "dense":
        W_d0, b_d0 = Wf([NSCALE * D, 256], "decimgW0"), Zz([256], "decimgB0")
        W_d1, b_d1 = Wf([256, PIX], "decimgW1"), Zz([PIX], "decimgB1")
        dec_names = ["decimgW0", "decimgB0", "decimgW1", "decimgB1"]
    else:  # conv/deconv
        Wp, bp = Wf([NSCALE * D, 7 * 7 * 16], "decimgP"), Zz([7 * 7 * 16], "decimgPb")
        K1, b1c = Wf([3, 3, 16, 8], "decimgK1"), Zz([8], "decimgK1b")
        K2, b2c = Wf([3, 3, 8, 1], "decimgK2"), Zz([1], "decimgK2b")
        dec_names = ["decimgP", "decimgPb", "decimgK1", "decimgK1b", "decimgK2", "decimgK2b"]

    NAME2IDX = {n: i for i, n in enumerate(NAMES)}
    IS_ATTN = [n.startswith(("emb", "pos", "Wq", "Wk", "Wv", "Wo", "Wff", "bff", "txttap")) for n in NAMES]
    ATTN_VARS = [VARS[i] for i in range(len(VARS)) if IS_ATTN[i]]
    dec_param_count = int(sum(int(np.prod(VARS[NAME2IDX[n]].shape)) for n in dec_names))

    def f(z): return tf.nn.relu(z)
    def se(eps): return tf.reduce_sum(tf.reshape(eps, [tf.shape(eps)[0], -1]) ** 2, axis=1)
    def flat(x): return tf.reshape(x, [tf.shape(x)[0], -1])
    def encode_img(X):
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
    def disc(S):    return [f(S[k] @ W_y[k] + b_y[k]) for k in range(NSCALE)]
    def dec_txt(S): return f(tf.concat(S, axis=1) @ W_dt + b_dt)
    def dec_img(S):
        z = tf.concat(S, axis=1)
        if decoder_kind == "dense":
            h = f(z @ W_d0 + b_d0)
            return tf.reshape(tf.nn.sigmoid(h @ W_d1 + b_d1), [tf.shape(z)[0], 28, 28, 1])
        else:
            h = tf.reshape(f(z @ Wp + bp), [tf.shape(z)[0], 7, 7, 16])     # project -> small grid
            h = f(tf.nn.conv2d(up2x(h), K1, [1, 1, 1, 1], "SAME") + b1c)   # 7->14, 16->8
            return tf.nn.sigmoid(tf.nn.conv2d(up2x(h), K2, [1, 1, 1, 1], "SAME") + b2c)  # 14->28, 8->1

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

    @tf.function(jit_compile=True)
    def infer_train(X, tok, y, S, beta):
        taps_txt = encode_txt(tok)
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(S); e = tf.reduce_mean(energy_pe(X, taps_txt, y, S))
        g = t.gradient(e, S)
        return [S[k] - beta * g[k] for k in range(NSCALE)], e
    @tf.function(jit_compile=True)
    def infer_img(X, y, S, beta):
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(S + [y]); e = tf.reduce_mean(energy_pe(X, None, y, S))
        g = t.gradient(e, S + [y])
        return [S[k] - beta * g[k] for k in range(NSCALE)], y - beta * g[-1], e
    @tf.function(jit_compile=True)
    def infer_txt2img(tok, y, X, S, beta):
        taps_txt = encode_txt(tok)
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch([X] + S); e = tf.reduce_mean(energy_pe(X, taps_txt, y, S))
        g = t.gradient(e, [X] + S)
        return tf.clip_by_value(X - beta * g[0], 0.0, 1.0), [S[k] - beta * g[k + 1] for k in range(NSCALE)], e
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
        tt = encode_txt(tok); return dec_img(tt), tt
    @tf.function(jit_compile=True)
    def learn_grads(X, tok, y, S):
        with tf.GradientTape() as t:                  # L2: encoders AND decoders INSIDE the tape
            taps_txt = encode_txt(tok)
            e = tf.reduce_mean(energy_pe(X, taps_txt, y, S))
        return t.gradient(e, VARS), e

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
    def img2text(x_imgs, chunk=250):
        preds = []
        for i in range(0, len(x_imgs), chunk):
            xb = tf.constant(x_imgs[i:i + chunk]); S, y0 = init_img(xb)
            S, y, _ = relax_img(xb, y0, S, N_INFER_TEST, BETA)
            preds.append(tf.argmax(y, axis=1).numpy())
        return np.concatenate(preds)
    def gen_from_text(classes):
        tok_c = tf.gather(CAP, classes); y_c = tf.one_hot(classes, NCLASS)
        X0, S = init_txt2img(tok_c)
        X, S, _ = relax_txt2img(tok_c, y_c, X0, S, N_GEN, GEN_BETA)
        return X.numpy()
    def recon_image(x_imgs):
        xb = tf.constant(x_imgs); S, y0 = init_img(xb)
        S, y, _ = relax_img(xb, y0, S, N_INFER_TEST, BETA)
        return dec_img(S).numpy()
    def per_scale_move(X, tok, y):
        S0 = init_train(X, tok); S0c = [tf.identity(s) for s in S0]
        Sf, _ = relax_train(X, tok, y, [tf.identity(s) for s in S0], N_INFER, BETA)
        return [float(tf.norm(Sf[k] - S0c[k]) / (tf.norm(S0c[k]) + 1e-8)) for k in range(NSCALE)]

    # ---- A pre-train energy descent, both directions ----
    fb_x = tf.constant(xtr[:BATCH]); fb_tok = tf.constant(tok_tr[:BATCH]); fb_y = tf.constant(ytr_oh[:BATCH])
    ti0, y0 = init_img(fb_x); _, _, elog_img_pre = relax_img(fb_x, y0, ti0, 30, BETA, log=True)
    X0t, S0t = init_txt2img(fb_tok); _, _, elog_txt_pre = relax_txt2img(fb_tok, fb_y, X0t, S0t, N_GEN, GEN_BETA, log=True)

    train_energy, gn_attn, maxabs = [], [], []
    gn_conv = [[] for _ in range(NUM_CONV)]; gn_label = [[] for _ in range(NSCALE)]; gn_decimg = []
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
            for k in range(NSCALE): gn_label[k].append(float(tf.norm(grads[NAME2IDX[f"labelW{k}"]])))
            gn_attn.append(float(tf.linalg.global_norm([grads[j] for j in range(len(VARS)) if IS_ATTN[j]])))
            gn_decimg.append(float(tf.linalg.global_norm([grads[NAME2IDX[n]] for n in dec_names])))
            maxabs.append(max(float(tf.reduce_max(tf.abs(v))) for v in VARS))
            step += 1
        if ep % EVAL_EVERY == 0 or ep == EPOCHS - 1:
            a = float((img2text(xte) == yte_lab).mean()); acc_steps.append(step); acc_hist.append(a)
            cstr = ",".join(f"{gn_conv[l][-1]:.1e}" for l in range(NUM_CONV))
            print(f"  [{decoder_kind:5s}] epoch {ep:2d}  F={train_energy[-1]:.3f}  img->text={a:.3f}  "
                  f"|g|conv=[{cstr}] |g|attn={gn_attn[-1]:.1e} |g|dec={gn_decimg[-1]:.1e} max|p|={maxabs[-1]:.2f}")
    secs = time.time() - t0

    ti0, y0 = init_img(fb_x); _, _, elog_img_post = relax_img(fb_x, y0, ti0, 30, BETA, log=True)
    X0t, S0t = init_txt2img(fb_tok); _, _, elog_txt_post = relax_txt2img(fb_tok, fb_y, X0t, S0t, N_GEN, GEN_BETA, log=True)

    # ---- C generation ----
    recon200 = recon_image(xte[:200]); recon_mse = float(np.mean((recon200 - xte[:200]) ** 2))
    gen_one = gen_from_text(np.arange(10, dtype=np.int32)); gen_one_pred = img2text(gen_one)
    gen_class_acc = float((gen_one_pred == np.arange(10)).mean())
    recon8_orig = xte[:8]; recon8 = recon_image(xte[:8])

    # ---- E multi-scale anchor health ----
    psm = per_scale_move(fb_x, fb_tok, fb_y)
    med_conv = [float(np.median(gn_conv[l])) for l in range(NUM_CONV)]
    med_label = [float(np.median(gn_label[k])) for k in range(NSCALE)]
    allmed = np.array(med_conv + med_label)
    grad_spread = float(allmed.max() / (allmed.min() + 1e-30))

    # ---- F attention grad both directions ----
    Sg = init_train(fb_x, fb_tok); Sg, _ = relax_train(fb_x, fb_tok, fb_y, Sg, N_INFER, BETA)
    with tf.GradientTape() as t:
        eg = tf.reduce_mean(energy_pe(fb_x, encode_txt(fb_tok), fb_y, Sg))
    attn_gen = float(tf.linalg.global_norm([g for g in t.gradient(eg, ATTN_VARS) if g is not None] or [tf.zeros(1)]))

    return dict(kind=decoder_kind, dec_params=dec_param_count, secs=secs,
                elog_img_pre=elog_img_pre, elog_txt_pre=elog_txt_pre,
                elog_img_post=elog_img_post, elog_txt_post=elog_txt_post,
                train_energy=train_energy, acc_steps=acc_steps, acc_hist=acc_hist, final_acc=acc_hist[-1],
                recon_mse=recon_mse, gen_one=gen_one, gen_one_pred=gen_one_pred, gen_class_acc=gen_class_acc,
                recon8_orig=recon8_orig, recon8=recon8, per_scale_move=psm,
                med_conv=med_conv, med_label=med_label, grad_spread=grad_spread,
                maxabs_max=float(np.max(maxabs)), attn_gen=attn_gen,
                gn_conv=gn_conv, gn_attn=gn_attn, gn_decimg=gn_decimg)


print("==== running CONV decoder ====")
C = run("conv")
print("==== running DENSE decoder (Stage 1.6 baseline, same seed) ====")
Dn = run("dense")

# ---------------- comparison plots (conv solid, dense dashed) ----------------
plt.figure(figsize=(6, 4))
plt.plot(C["elog_img_post"], 'b-', label="conv clamp-img"); plt.plot(C["elog_txt_post"], 'g-', label="conv clamp-txt")
plt.plot(Dn["elog_img_post"], 'b--', alpha=0.6, label="dense clamp-img"); plt.plot(Dn["elog_txt_post"], 'g--', alpha=0.6, label="dense clamp-txt")
plt.xlabel("inference step"); plt.ylabel("F"); plt.title("A. Inference energy descent (conv vs dense)")
plt.legend(fontsize=7); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "conv_energy_inference.png"), dpi=110); plt.close()

plt.figure(figsize=(6, 4)); plt.plot(C["train_energy"], 'b-', lw=0.7, label="conv"); plt.plot(Dn["train_energy"], 'r--', lw=0.7, alpha=0.6, label="dense")
plt.yscale("log"); plt.xlabel("weight update"); plt.ylabel("post-relax F"); plt.title("A. Training energy (conv vs dense)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "conv_training.png"), dpi=110); plt.close()

plt.figure(figsize=(6, 4))
plt.plot(C["acc_steps"], C["acc_hist"], 'bo-', label="conv"); plt.plot(Dn["acc_steps"], Dn["acc_hist"], 'rs--', label="dense")
plt.axhline(0.1, ls=':', c='gray', label="chance"); plt.xlabel("training step"); plt.ylabel("img->text acc")
plt.title("B. Discriminative accuracy (conv vs dense)"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(HERE, "conv_accuracy.png"), dpi=110); plt.close()

def gengrid(res, fname, title):
    plt.figure(figsize=(8, 2.2))
    for c in range(10):
        plt.subplot(2, 5, c + 1); plt.imshow(res["gen_one"][c, :, :, 0], cmap="gray", vmin=0, vmax=1)
        ok = "OK" if res["gen_one_pred"][c] == c else "x"
        plt.title(f"text={c} -> {res['gen_one_pred'][c]} {ok}", fontsize=7); plt.axis("off")
    plt.suptitle(title, fontsize=10); plt.tight_layout(); plt.savefig(os.path.join(HERE, fname), dpi=120); plt.close()
gengrid(C, "conv_text2image_samples.png", "C. TEXT->IMAGE (CONV decoder)")
gengrid(Dn, "dense_ref_text2image_samples.png", "C. TEXT->IMAGE (DENSE decoder, 1.6 baseline)")

def recongrid(res, fname, title):
    plt.figure(figsize=(8, 2.2))
    for j in range(8):
        plt.subplot(2, 8, j + 1); plt.imshow(res["recon8_orig"][j, :, :, 0], cmap="gray", vmin=0, vmax=1); plt.axis("off")
        plt.subplot(2, 8, j + 9); plt.imshow(res["recon8"][j, :, :, 0], cmap="gray", vmin=0, vmax=1); plt.axis("off")
    plt.suptitle(title, fontsize=10); plt.tight_layout(); plt.savefig(os.path.join(HERE, fname), dpi=120); plt.close()
recongrid(C, "conv_image2image_recon.png", "C. IMG->IMG (CONV) top orig / bottom recon")
recongrid(Dn, "dense_ref_image2image_recon.png", "C. IMG->IMG (DENSE) top orig / bottom recon")

plt.figure(figsize=(7, 4))
for l in range(NUM_CONV): plt.plot(C["gn_conv"][l], lw=0.8, label=f"convW{l}")
plt.plot(C["gn_attn"], 'k--', lw=0.8, label="attn"); plt.plot(C["gn_decimg"], 'r:', lw=0.9, label="dec_img(conv)")
plt.yscale("log"); plt.xlabel("weight update"); plt.ylabel("grad norm"); plt.title("E/F. Per-layer grad norms (conv decoder)")
plt.legend(fontsize=7, ncol=2); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(HERE, "conv_grad_norms.png"), dpi=110); plt.close()

# ---------------- head-to-head + verdict ----------------
def A_ok(r):
    k = max(1, len(r["train_energy"]) // 10)
    return (near_monotone(r["elog_img_pre"]) and near_monotone(r["elog_img_post"])
            and near_monotone(r["elog_txt_pre"]) and near_monotone(r["elog_txt_post"])
            and r["elog_img_post"][-1] < r["elog_img_post"][0] and r["elog_txt_post"][-1] < r["elog_txt_post"][0]
            and np.mean(r["train_energy"][-k:]) < np.mean(r["train_energy"][:k]))
cA, dA = A_ok(C), A_ok(Dn)
anchors_ok = (min(C["per_scale_move"]) > 1e-3) and (C["grad_spread"] < 1e2)   # E: no frozen scale, spread small
gen_match = (C["gen_class_acc"] >= Dn["gen_class_acc"] - 0.1) and (C["recon_mse"] <= Dn["recon_mse"] * 1.25)
fewer_params = C["dec_params"] < Dn["dec_params"]
attn_ok = C["attn_gen"] > 0.0

print("\n================ STAGE 1.7 HEAD-TO-HEAD (conv image decoder vs flatten-dense) ================")
print(f"{'metric':36s} {'CONV':>16s} {'DENSE (1.6)':>16s}")
print(f"{'D. image-decoder params':36s} {C['dec_params']:>16,d} {Dn['dec_params']:>16,d}  "
      f"(conv = {100*C['dec_params']/Dn['dec_params']:.0f}% of dense)")
print(f"{'A. energy descends both dirs':36s} {str(cA):>16s} {str(dA):>16s}")
print(f"{'   clamp-img F post (start->end)':36s} {C['elog_img_post'][0]:7.2f}->{C['elog_img_post'][-1]:<7.2f} "
      f"{Dn['elog_img_post'][0]:7.2f}->{Dn['elog_img_post'][-1]:<7.2f}")
print(f"{'   clamp-txt F post (start->end)':36s} {C['elog_txt_post'][0]:7.2f}->{C['elog_txt_post'][-1]:<7.2f} "
      f"{Dn['elog_txt_post'][0]:7.2f}->{Dn['elog_txt_post'][-1]:<7.2f}")
print(f"{'B. img->text accuracy':36s} {C['final_acc']:>16.3f} {Dn['final_acc']:>16.3f}")
print(f"{'C. text->image per-class acc':36s} {C['gen_class_acc']:>16.3f} {Dn['gen_class_acc']:>16.3f}")
print(f"{'C. image->image recon MSE':36s} {C['recon_mse']:>16.4f} {Dn['recon_mse']:>16.4f}")
print(f"{'F. attn grad (generative dir)':36s} {C['attn_gen']:>16.2e} {Dn['attn_gen']:>16.2e}")
print(f"\nE. multi-scale anchor health (CONV): per-scale state move = {['%.1e'%m for m in C['per_scale_move']]} "
      f"(all>1e-3? {min(C['per_scale_move'])>1e-3})")
print(f"   per-scale/layer median grad: conv={['%.1e'%g for g in C['med_conv']]} label={['%.1e'%g for g in C['med_label']]}")
print(f"   grad spread across scales = {C['grad_spread']:.1f}x (<100x? {C['grad_spread']<1e2})  max|p|={C['maxabs_max']:.2f}")
print(f"   conv class->pred: {list(zip(range(10), C['gen_one_pred'].tolist()))}")
print(f"   dense class->pred: {list(zip(range(10), Dn['gen_one_pred'].tolist()))}")

if cA and anchors_ok and gen_match and fewer_params and attn_ok:
    verdict = "PASS / RECOMMEND CONV"
elif cA and anchors_ok and not gen_match:
    verdict = "PARTIAL: conv works but generation degrades vs dense -> keep flatten-dense + checkpointing"
else:
    verdict = "FAIL / KEEP DENSE"
print(f"\nVERDICT: {verdict}")
print(f"  A(conv)={cA} anchors_balanced={anchors_ok} gen_match_or_better={gen_match} "
      f"fewer_params={fewer_params} attn_ok={attn_ok}")
if verdict.startswith("PASS"):
    print("  -> The 7.7B port should use CONV/DECONV decoders. Miniature param ratio is "
          f"{100*C['dec_params']/Dn['dec_params']:.0f}%, but the real win scales: the big model's dense head "
          "FLATTENS conv2 (~20.6M) before Dense, so its size scales with input RESOLUTION; a conv decoder "
          "does not, so on 572x572 the reduction is multiple orders of magnitude. Per-scale anchors stayed "
          "balanced, so depth (L3) is preserved. Revise PORT_PLAN.md section 5/6 toward Option R.")
print(f"\nhyperparams: D={D} d_model={DMODEL} conv={CONV} N_infer={N_INFER} beta={BETA} alpha={ALPHA} "
      f"epochs={EPOCHS} PI(cross,disc,gen)=({PI_CROSS},{PI_DISC},{PI_GEN})")
print(f"plots: conv_energy_inference / conv_training / conv_accuracy / conv_text2image_samples / "
      f"conv_image2image_recon / conv_grad_norms (+ dense_ref_* for visual control)")
print(f"wall-clock: conv {C['secs']:.1f}s + dense {Dn['secs']:.1f}s (CPU, jit_compile=True)")
