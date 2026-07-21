"""Objective diagnostic (read-only-ish, throwaway): does the phase-2 generative weight step
SHRINK the decode weights via weight decay? A/B test: run N generative steps from a clean
recon start with weight decay ON (3e-2) vs OFF (0) on the decode layers, and compare the
decode weight-norm change and the generated-image brightness.

Hypothesis: when the bridge error is small, the LARS step wts -= lr*trust*(g + wd*wts) is
dominated by the -lr*wd*wts shrinkage term, so the decode weights (hence output amplitude)
decay over generative steps. If wd=OFF stops the shrinkage and keeps brightness up, the fix
is to disable weight decay for the generative weight step.
"""
import argparse
import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import COCO64_GEN as C
from conv_pcn_layer import Conv2DPCNLayer
import coco64_data as D
from train_coco64 import generative_step

CLIP = 400.0


def decode_set(m):
    lat = set()
    for a_, b_ in m._shared_latent_pairs:
        lat.add(id(a_)); lat.add(id(b_))
    return [L for L in m._image_path_layers
            if hasattr(L, "update_wts") and id(L) not in lat and L is not m.img_input]


def decode_wnorm(decode):
    tot = 0.0
    for L in decode:
        w = getattr(L, "wts", None)
        if w is not None:
            tot += float(tf.reduce_sum(tf.square(w)))
    return tot ** 0.5


def build_restore(ckpt, img, txt, mask, decode_wd):
    m = EncoderEncoderPCN(1e-4, config=C)
    for L in m.trainable_layers:
        if hasattr(L, "state_clip"):
            L.state_clip = CLIP
        if isinstance(L, Conv2DPCNLayer) and getattr(L, "stride", 1) == 1:
            L.activation = "gelu"
        if hasattr(L, "weight_decay"):
            L.weight_decay = 3e-2
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts", "b", "gamma", "beta")
             if isinstance(getattr(l, at, None), tf.Variable)]
    ck = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    latest = tf.train.latest_checkpoint(ckpt); assert latest
    ck.restore(latest).expect_partial()
    dec = decode_set(m)
    for L in dec:                      # the arm's weight decay on the decode layers (the gen-step target set)
        if hasattr(L, "weight_decay"):
            L.weight_decay = decode_wd
    return m, dec


def gen_brightness(m, img, txt, mask):
    # text-driven boosted generation (gamma 1.0), report output mean
    chain, L = [], m._infonce_codes[0]
    while L is not None:
        chain.append(L); L = getattr(L, "prev_layer", None)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(tf.zeros_like(img), txt, mask)
    m.img_input.is_clamped = False
    for _ in range(150):
        for L in m.trainable_layers:
            L.update_state()
        m.img_input.update_state()
        for L in chain:
            st = getattr(L, "state", None)
            if st is None or not L.next_layers:
                continue
            nxt = L.next_layers[0]
            if not hasattr(nxt, "predict_prev"):
                continue
            td = nxt.predict_prev()
            if td.shape != st.shape:
                continue
            st.assign(tf.clip_by_value(st + 1.0 * (td - st), -CLIP, CLIP))
    return float(m.img_input.predict_next().numpy().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_gen_best")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    img, txt, mask = D.load_batch(2000, seed=0)
    img = np.asarray(img[:a.k], np.float32); txt = np.asarray(txt[:a.k], np.float32); mask = np.asarray(mask[:a.k], np.float32)
    T = tf.convert_to_tensor
    print(f"ckpt={a.ckpt} steps={a.steps} k={a.k} (A/B weight decay ON vs OFF for the generative step)", flush=True)
    print(f"TRUE image mean={img.mean():.4f}", flush=True)

    for wd in (3e-2, 0.0):
        m, dec = build_restore(a.ckpt, T(img), T(txt), T(mask), wd)
        w0 = decode_wnorm(dec)
        b0 = gen_brightness(m, T(img), T(txt), T(mask))
        for _ in range(a.steps):
            generative_step(m, img, txt, mask, 15, 15, 3e-4)
        w1 = decode_wnorm(dec)
        b1 = gen_brightness(m, T(img), T(txt), T(mask))
        print(f"[wd={wd:g}] decode ||W|| {w0:.2f} -> {w1:.2f} (ratio {w1/w0:.3f}); gen mean {b0:.4f} -> {b1:.4f}", flush=True)
    print("OBJECTIVE_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
