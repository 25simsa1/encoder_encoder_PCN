"""Cross-modal ceiling probe: how much of the IMAGE-set latent is linearly recoverable
from the text path, per tap, under different X choices: (a) text inter after relax,
(b) text inter at pass-through (no relax), (c) the wider text dense_relu beneath it at
pass-through. R2 only, no assignment. Throwaway instrument."""
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


def fit_r2(X, Y, lam):
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    mx, my = X.mean(0, keepdims=True), Y.mean(0, keepdims=True)
    Xc, Yc = X - mx, Y - my
    if X.shape[1] <= X.shape[0]:
        G = Xc.T @ Xc + lam * np.eye(X.shape[1])
        W = np.linalg.solve(G, Xc.T @ Yc)
        pred = Xc @ W
    else:
        G = Xc @ Xc.T + lam * np.eye(X.shape[0])
        pred = Xc @ (Xc.T @ np.linalg.solve(G, Yc))
    return 1.0 - float(((Yc - pred) ** 2).sum() / ((Yc ** 2).sum() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="ckpt_wide5_best")
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--relax", type=int, default=15)
    ap.add_argument("--lam", type=float, default=10.0)
    a = ap.parse_args()
    img, txt, mask = D.load_batch(a.pairs, seed=0)
    T = tf.convert_to_tensor
    m = EncoderEncoderPCN(1e-4, config=C)
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
    ck.restore(tf.train.latest_checkpoint(a.base_ckpt)).expect_partial()
    print(f"restored {a.base_ckpt}", flush=True)

    pairs = m._shared_latent_pairs
    NT = len(pairs)
    Y = {i: [] for i in range(NT)}
    XA = {i: [] for i in range(NT)}   # inter, relaxed
    XB = {i: [] for i in range(NT)}   # inter, pass-through
    XC = {i: [] for i in range(NT)}   # dense_relu beneath, pass-through
    n = a.pairs
    for s in range(0, n, 8):
        bi = slice(s, s + 8)
        ib = T(np.asarray(img[bi], np.float32)); tb = T(np.asarray(txt[bi], np.float32))
        mb = T(np.asarray(mask[bi], np.float32))
        m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(ib, tb, mb)
        relax(m, a.relax)
        for i, (Limg, Ltxt) in enumerate(pairs):
            Y[i].append(Limg.state.numpy())
        m.img_input.is_clamped = False; m.txt_input.is_clamped = True
        m.pass_through(ib * 0.0, tb, mb)
        for i, (Limg, Ltxt) in enumerate(pairs):
            XB[i].append(Ltxt.prev_layer.state.numpy())
            deeper = Ltxt.prev_layer.prev_layer
            st = deeper.state.numpy()
            XC[i].append(st.reshape(st.shape[0], -1))
        relax(m, a.relax)
        for i, (Limg, Ltxt) in enumerate(pairs):
            XA[i].append(Ltxt.prev_layer.state.numpy())
        if s % 400 == 0:
            print(f"collected {s + 8}/{n}", flush=True)

    for i in range(NT):
        y = np.concatenate(Y[i], 0)
        ra = fit_r2(np.concatenate(XA[i], 0), y, a.lam)
        rb = fit_r2(np.concatenate(XB[i], 0), y, a.lam)
        rc = fit_r2(np.concatenate(XC[i], 0), y, a.lam)
        print(f"tap {i}: R2 inter-relaxed={ra:.4f}  inter-passthru={rb:.4f}  dense-passthru={rc:.4f}", flush=True)
    print("TEXT_CEILING_DONE", flush=True)


if __name__ == "__main__":
    main()
