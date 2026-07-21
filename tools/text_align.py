"""Fix B: text-to-code alignment by closed-form ridge on the text path's final edges.
For each shared-latent tap, fit the text edge (text inter -> shared latent) so the
caption-driven forward lands on the IMAGE-set latent for the same pair: the exact
least-squares optimum of the same local cross-modal PC objective at the shared latent.
X = text-inter states under caption-clamped relax (image free),
Y = shared-latent states under recon-clamp relax (the latents that decode at 0.0420).
Assigns wts/b of the text latent edges and saves a full checkpoint. Throwaway instrument."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="ckpt_wide5_best")
    ap.add_argument("--out-ckpt", default="ckpt_wide5_ta")
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

    # (image_latent, text_latent) aliased pairs; the text edge maps its prev (text inter) -> latent
    pairs = m._shared_latent_pairs
    print(f"{len(pairs)} shared-latent pairs", flush=True)
    Xs = {i: [] for i in range(len(pairs))}
    Ys = {i: [] for i in range(len(pairs))}
    n = a.pairs
    for s in range(0, n, 8):
        bi = slice(s, s + 8)
        ib = T(np.asarray(img[bi], np.float32)); tb = T(np.asarray(txt[bi], np.float32))
        mb = T(np.asarray(mask[bi], np.float32))
        # capture Y: recon clamp (both), relax -> the latents that decode at 0.0420
        m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(ib, tb, mb)
        relax(m, a.relax)
        for i, (Limg, Ltxt) in enumerate(pairs):
            Ys[i].append(Limg.state.numpy())
        # capture X: caption clamped, image free, relax -> the text-inter states feeding each latent
        m.img_input.is_clamped = False; m.txt_input.is_clamped = True
        m.pass_through(ib * 0.0, tb, mb)
        relax(m, a.relax)
        for i, (Limg, Ltxt) in enumerate(pairs):
            Xs[i].append(Ltxt.prev_layer.state.numpy())
        if s % 400 == 0:
            print(f"collected {s + 8}/{n}", flush=True)

    for i, (Limg, Ltxt) in enumerate(pairs):
        X = np.concatenate(Xs[i], 0).astype(np.float64)
        Y = np.concatenate(Ys[i], 0).astype(np.float64)
        mx, my = X.mean(0, keepdims=True), Y.mean(0, keepdims=True)
        Xc, Yc = X - mx, Y - my
        G = Xc.T @ Xc + a.lam * np.eye(X.shape[1])
        W = np.linalg.solve(G, Xc.T @ Yc)                      # (inter_dim, latent_dim)
        b = (my - mx @ W).reshape(-1)
        pred = Xc @ W
        r2 = 1.0 - float(((Yc - pred) ** 2).sum() / ((Yc ** 2).sum() + 1e-9))
        act = getattr(Ltxt, "activation_name", getattr(Ltxt, "activation", "?"))
        Ltxt.wts.assign(tf.convert_to_tensor(W, tf.float32))
        Ltxt.b.assign(tf.convert_to_tensor(b, tf.float32))
        print(f"text edge {i} (act={act}): X{X.shape} -> Y{Y.shape}  R2={r2:.4f}", flush=True)

    out = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    mgr = tf.train.CheckpointManager(out, a.out_ckpt, max_to_keep=1)
    mgr.save()
    print(f"saved {a.out_ckpt}", flush=True)
    print("TEXT_ALIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
