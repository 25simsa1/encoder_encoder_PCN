"""Localize relaxation instability: forward pass_through, then N relax steps,
printing per-image-path-layer state RMS each step. Compare a stable narrow
encoder against an exploding wide one. Throwaway instrument."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--clip", type=float, default=0.0)   # 0 = UNclipped (expose the raw dynamics)
    a = ap.parse_args()
    img, txt, mask = D.load_batch(8, seed=0)
    T = tf.convert_to_tensor
    m = EncoderEncoderPCN(1e-4, config=C)
    for L in m.trainable_layers:
        if hasattr(L, "state_clip") and a.clip > 0:
            L.state_clip = a.clip
        if isinstance(L, Conv2DPCNLayer) and getattr(L, "stride", 1) == 1:
            L.activation = "gelu"
    m.img_input.is_clamped = True
    m.txt_input.is_clamped = True
    m.pass_through(T(np.asarray(img, np.float32)), T(np.asarray(txt, np.float32)),
                   T(np.asarray(mask, np.float32)))
    ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts", "b", "gamma", "beta")
             if isinstance(getattr(l, at, None), tf.Variable)]
    ck = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    ck.restore(tf.train.latest_checkpoint(a.ckpt)).expect_partial()
    # re-run the forward pass with the restored weights
    m.pass_through(T(np.asarray(img, np.float32)), T(np.asarray(txt, np.float32)),
                   T(np.asarray(mask, np.float32)))
    layers = m._image_path_layers
    names = [f"{i:02d}:{type(L).__name__[:5]}" for i, L in enumerate(layers)]

    def rms(L):
        s = L.state
        return float(tf.sqrt(tf.reduce_mean(tf.square(s)))) if s is not None else 0.0

    base = [rms(L) for L in layers]
    print("layer            fwdRMS  " + "".join(f"  s{k+1:<7d}" for k in range(a.steps)), flush=True)
    hist = [[] for _ in layers]
    for _ in range(a.steps):
        for L in m.trainable_layers:
            L.update_state()
        for i, L in enumerate(layers):
            hist[i].append(rms(L))
    for i, nm in enumerate(names):
        row = "".join(f"{v:9.2f}" for v in hist[i])
        print(f"{nm:15s}{base[i]:9.2f}{row}", flush=True)
    print("RELAX_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
