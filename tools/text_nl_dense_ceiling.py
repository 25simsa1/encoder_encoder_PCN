"""The clean confirmatory caption-to-image probe. The earlier nonlinear probe used the
deepest COUPLED code, which had collapsed (near-identical directions) -> artifact. Here we
fit from the UNCOLLAPSED wide text dense feature (Ltxt.prev_layer.prev_layer, the one that
scored ~0.19 LINEARLY in text_ceiling) to the image-set latent, LINEAR ridge vs RBF kernel
ridge, 80/20 train/test. Standardize features (no unit-normalize) so the RBF is well-scaled.
High RBF TRAIN R2 => a nonlinear readout can render captions on the overfit (I was wrong to
close it). RBF ~0 on TRAIN too => the info is not in the text representation even nonlinearly
= representational limit confirmed. Runs on both the recon encoder and the coupled one."""
import argparse
import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
import os
from pcn_config import COCO64_GEN, COCO64_WIDE
C = COCO64_WIDE if os.environ.get('PCN_TOOL_CONFIG') == 'coco64_wide' else COCO64_GEN
from conv_pcn_layer import Conv2DPCNLayer
import coco64_data as D
CLIP = 400.0


def relax(m, k):
    for _ in range(k):
        for L in m.trainable_layers:
            L.update_state()


def r2_linear(Xtr, Ytr, Xte, Yte, lam):
    mx, my = Xtr.mean(0, keepdims=True), Ytr.mean(0, keepdims=True)
    Xc, Yc = Xtr - mx, Ytr - my
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
    def r2(X, Y):
        p = (X - mx) @ W + my
        return 1.0 - ((Y - p) ** 2).sum() / (((Y - my) ** 2).sum() + 1e-9)
    return r2(Xtr, Ytr), r2(Xte, Yte)


def r2_rbf(Xtr, Ytr, Xte, Yte, lam, gamma):
    def kern(A, B):
        d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(d, 0))
    my = Ytr.mean(0, keepdims=True)
    Ktr = kern(Xtr, Xtr)
    alpha = np.linalg.solve(Ktr + lam * np.eye(Xtr.shape[0]), Ytr - my)
    def r2(X, Y):
        p = kern(X, Xtr) @ alpha + my
        return 1.0 - ((Y - p) ** 2).sum() / (((Y - my) ** 2).sum() + 1e-9)
    return r2(Xtr, Ytr), r2(Xte, Yte)


def run(ckpt, pairs, relax_k, lam):
    img, txt, mask = D.load_batch(pairs, seed=0)
    T = tf.convert_to_tensor
    m = EncoderEncoderPCN(1e-3, config=C)
    for L in m.trainable_layers:
        if hasattr(L, "state_clip"):
            L.state_clip = CLIP
        if isinstance(L, Conv2DPCNLayer) and getattr(L, "stride", 1) == 1:
            L.activation = "gelu"
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(T(np.asarray(img[:8], np.float32)), T(np.asarray(txt[:8], np.float32)),
                   T(np.asarray(mask[:8], np.float32)))
    ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts", "b", "gamma", "beta")
             if isinstance(getattr(l, at, None), tf.Variable)]
    ck = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    ck.restore(tf.train.latest_checkpoint(ckpt)).expect_partial()
    print(f"=== {ckpt}", flush=True)
    pairsL = m._shared_latent_pairs
    NT = len(pairsL)
    Y = {i: [] for i in range(NT)}
    Xd = {i: [] for i in range(NT)}
    n = pairs
    for s in range(0, n, 8):
        bi = slice(s, s + 8)
        ib = T(np.asarray(img[bi], np.float32)); tb = T(np.asarray(txt[bi], np.float32))
        mb = T(np.asarray(mask[bi], np.float32))
        m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(ib, tb, mb)
        relax(m, relax_k)
        for i, (Li, Lt) in enumerate(pairsL):
            Y[i].append(Li.state.numpy())
        m.img_input.is_clamped = False; m.txt_input.is_clamped = True
        m.pass_through(ib * 0.0, tb, mb)   # pass-through, caption clamped, image free
        for i, (Li, Lt) in enumerate(pairsL):
            st = Lt.prev_layer.prev_layer.state.numpy()   # the wide uncollapsed text feature
            Xd[i].append(st.reshape(st.shape[0], -1))
        if s % 400 == 0:
            print(f"collected {s + 8}/{n}", flush=True)

    ntr = int(0.8 * n)
    for i in range(NT):
        X = np.concatenate(Xd[i], 0).astype(np.float64)
        Yi = np.concatenate(Y[i], 0).astype(np.float64)
        # standardize features (per-dim), so RBF distances are well-scaled
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
        Xs = (X - mu) / sd
        Xtr, Xte, Ytr, Yte = Xs[:ntr], Xs[ntr:], Yi[:ntr], Yi[ntr:]
        sub = Xtr[:400]
        d = (sub * sub).sum(1)[:, None] + (sub * sub).sum(1)[None, :] - 2 * sub @ sub.T
        med = np.median(d[d > 0]); gamma = 1.0 / (med + 1e-9)
        ltr, lte = r2_linear(Xtr, Ytr, Xte, Yte, lam)
        rtr, rte = r2_rbf(Xtr, Ytr, Xte, Yte, lam, gamma)
        print(f"tap {i} (featdim={X.shape[1]}, gamma={gamma:.3e}): "
              f"linear train={ltr:.4f} test={lte:.4f} | RBF train={rtr:.4f} test={rte:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=["ckpt_wide5_best", "ckpt_wnce32_best"])
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--relax", type=int, default=15)
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    for c in a.ckpts:
        run(c, a.pairs, a.relax, a.lam)
    print("TEXT_NL_DENSE_DONE", flush=True)


if __name__ == "__main__":
    main()
