"""Per-edge differential-gain probe: is each top-down edge a real INVERSE or a mean-emitter?
For two different data batches A and B, compare the edge's top-down prediction DIFFERENTIAL
(pred_A - pred_B) against the true below-state differential (targ_A - targ_B). A healthy
inverse has gain ~1 and cosine ~1; a mean-emitter has gain ~0. Throwaway."""
import argparse
import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import COCO64_GEN as C
from conv_pcn_layer import Conv2DPCNLayer
import coco64_data as D
CLIP = 400.0


def build(ckpt, img, txt, mask, td_ckpt, td_affine):
    m = EncoderEncoderPCN(1e-4, config=C)
    for L in m.trainable_layers:
        if hasattr(L, "state_clip"):
            L.state_clip = CLIP
        if isinstance(L, Conv2DPCNLayer) and getattr(L, "stride", 1) == 1:
            L.activation = "gelu"
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts", "b", "gamma", "beta")
             if isinstance(getattr(l, at, None), tf.Variable)]
    ck = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    ck.restore(tf.train.latest_checkpoint(ckpt)).expect_partial()
    for L in m._image_path_layers:
        if hasattr(L, "enable_untied") and getattr(L, "wts", None) is not None:
            L.enable_untied()
    if td_affine:
        TD = [v for L in m._image_path_layers if getattr(L, "untied", False) for v in (L.wts_td, L.c_td)]
    else:
        TD = [L.wts_td for L in m._image_path_layers if getattr(L, "untied", False)]
    tck = tf.train.Checkpoint(**{f"t{i}": v for i, v in enumerate(TD)})
    tck.restore(tf.train.latest_checkpoint(td_ckpt)).expect_partial()
    print(f"restored {ckpt} + td (affine={td_affine})", flush=True)
    return m


def snap(m, img, txt, mask, decode):
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    for _ in range(15):
        for L in m.trainable_layers:
            L.update_state()
    out = {}
    for L in decode:
        P = getattr(L, "prev_layer", None)
        if P is None:
            continue
        try:
            out[id(L)] = (tf.identity(L.predict_prev()), tf.identity(P.predict_next()))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--td-affine", action="store_true")
    a = ap.parse_args()
    img, txt, mask = D.load_batch(2000, seed=0)
    A = slice(0, 8); B = slice(8, 16)
    T = tf.convert_to_tensor
    iA, tA, mA = T(np.asarray(img[A], np.float32)), T(np.asarray(txt[A], np.float32)), T(np.asarray(mask[A], np.float32))
    iB, tB, mB = T(np.asarray(img[B], np.float32)), T(np.asarray(txt[B], np.float32)), T(np.asarray(mask[B], np.float32))
    m = build(a.ckpt, iA, tA, mA, a.ckpt + "_td", a.td_affine)
    latent_ids = set()
    for a_, b_ in m._shared_latent_pairs:
        latent_ids.add(id(a_)); latent_ids.add(id(b_))
    decode = [L for L in m._image_path_layers
              if hasattr(L, "update_wts") and id(L) not in latent_ids and L is not m.img_input]
    names = {id(L): f"{i:02d}:{type(L).__name__}" for i, L in enumerate(m._image_path_layers)}
    SA = snap(m, iA, tA, mA, decode)
    SB = snap(m, iB, tB, mB, decode)
    print("=== per-edge differential gain (|dpred|/|dtarg|) and cosine, healthy inverse ~ (1, 1) ===", flush=True)
    for L in decode:
        if id(L) not in SA or id(L) not in SB:
            continue
        dp = SA[id(L)][0] - SB[id(L)][0]
        dt = SA[id(L)][1] - SB[id(L)][1]
        ndp = float(tf.norm(dp)); ndt = float(tf.norm(dt)) + 1e-9
        cos = float(tf.reduce_sum(dp * dt)) / (ndp * ndt + 1e-9)
        print(f"{names[id(L)]:>24s}  gain={ndp/ndt:8.4f}  cos={cos:7.4f}  |dtarg|={ndt:9.2f}", flush=True)
    print("EDGE_GAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
