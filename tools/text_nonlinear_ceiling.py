"""Decisive caption-to-image feasibility probe. The coupled text code is perfectly
discriminative (InfoNCE acc 1.0) yet the image latent is ~0 LINEARLY recoverable from it.
This tests whether a NONLINEAR readout closes the gap: fit RBF kernel ridge from the text
deepest code -> each image-set latent tap and compare R2 against a linear ridge baseline on
the SAME split (train/test 80/20, so we separate memorization from generalization).
High nonlinear TRAIN R2 => a nonlinear PC readout can render captions on the overfit.
High nonlinear TEST R2 => it also generalizes. Throwaway instrument."""
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
    # RBF kernel ridge: alpha = (K + lam I)^-1 Ytr ; predict K(x, Xtr) alpha
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_wnce32_best")
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--relax", type=int, default=15)
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    img, txt, mask = D.load_batch(a.pairs, seed=0)
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
    ck.restore(tf.train.latest_checkpoint(a.ckpt)).expect_partial()
    print(f"restored {a.ckpt}", flush=True)

    pairs = m._shared_latent_pairs
    ui, vi = m._infonce_codes
    NT = len(pairs)
    Y = {i: [] for i in range(NT)}
    Xcode = []
    n = a.pairs
    for s in range(0, n, 8):
        bi = slice(s, s + 8)
        ib = T(np.asarray(img[bi], np.float32)); tb = T(np.asarray(txt[bi], np.float32))
        mb = T(np.asarray(mask[bi], np.float32))
        m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(ib, tb, mb)
        relax(m, a.relax)
        for i, (Limg, Lt) in enumerate(pairs):
            Y[i].append(Limg.state.numpy())
        m.img_input.is_clamped = False; m.txt_input.is_clamped = True
        m.pass_through(ib * 0.0, tb, mb)
        relax(m, a.relax)
        Xcode.append(vi.state.numpy())          # the discriminative text deepest code
        if s % 400 == 0:
            print(f"collected {s + 8}/{n}", flush=True)

    X = np.concatenate(Xcode, 0).astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # unit codes (InfoNCE space)
    ntr = int(0.8 * X.shape[0])
    # median pairwise sq-dist for the RBF bandwidth
    sub = X[:400]
    d = (sub * sub).sum(1)[:, None] + (sub * sub).sum(1)[None, :] - 2 * sub @ sub.T
    med = np.median(d[d > 0]); gamma = 1.0 / (med + 1e-9)
    print(f"N={X.shape[0]} code_dim={X.shape[1]} train={ntr} rbf_gamma={gamma:.4f}", flush=True)
    for i in range(NT):
        Yi = np.concatenate(Y[i], 0).astype(np.float64)
        Xtr, Xte, Ytr, Yte = X[:ntr], X[ntr:], Yi[:ntr], Yi[ntr:]
        ltr, lte = r2_linear(Xtr, Ytr, Xte, Yte, a.lam)
        rtr, rte = r2_rbf(Xtr, Ytr, Xte, Yte, a.lam, gamma)
        print(f"tap {i}: linear R2 train={ltr:.4f} test={lte:.4f} | RBF R2 train={rtr:.4f} test={rte:.4f}", flush=True)
    print("TEXT_NONLINEAR_CEILING_DONE", flush=True)


if __name__ == "__main__":
    main()
