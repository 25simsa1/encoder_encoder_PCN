"""Darkness / output-scale diagnostic (read-only, fixed weights): why is text-driven
generation dark while reconstruction is bright? Localize where the scale is lost.

- Regime I (image-set latent): both inputs clamped (recon), relax; read the 5 shared-latent
  norms/stds. This is the latent the bright reconstruction decodes from.
- Regime T (text-set latent): caption clamped, image = zeros unclamped, relax with the
  top-down boost (gamma 1.0, matching the generation that moves off the blob); read the 5
  shared-latent norms/stds AND the generated-image mean/std/min/max.
- Compare: are text-set latents attenuated vs image-set (norm ratio << 1 => text under-drives
  the latent)? Is the generated image mean << the true image mean, and by how much (darkness
  quantified)? If latents match but output is dark, the decode attenuates; if latents are
  attenuated, the text path under-drives. Throwaway.
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

CLIP = 400.0


def build_restore(ckpt, img, txt, mask, weight_norm=False, wn_ckpt=None):
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
    latest = tf.train.latest_checkpoint(ckpt); assert latest, f"no ckpt in {ckpt}"
    ck.restore(latest).expect_partial(); print(f"restored {latest}", flush=True)
    if weight_norm:
        # same order as train_coco64: base weights are restored, THEN enable (g_mag from the
        # restored ||wts||, a placeholder), THEN restore the trained g_mag over it.
        nwn = 0
        for L in m.trainable_layers:
            if hasattr(L, "enable_weight_norm") and getattr(L, "wts", None) is not None:
                L.enable_weight_norm(); nwn += 1
        WN_W = [L.g_mag for L in m.trainable_layers if getattr(L, "weight_norm", False)]
        wn_ck = tf.train.Checkpoint(**{f"g{i}": v for i, v in enumerate(WN_W)})
        wl = tf.train.latest_checkpoint(wn_ckpt); assert wl, f"no wn ckpt in {wn_ckpt}"
        wn_ck.restore(wl).expect_partial(); print(f"restored wn {wl} on {nwn} layers", flush=True)
    return m


def latent_stats(m):
    out = []
    for a_, _ in m._shared_latent_pairs:   # image-side dense of each aliased pair
        s = a_.state
        n = float(tf.reduce_mean(tf.norm(tf.reshape(s, (s.shape[0], -1)), axis=1)))
        out.append((n, float(tf.math.reduce_std(s))))
    return out


def decode_chain(ui):
    chain, L = [], ui
    while L is not None:
        chain.append(L); L = getattr(L, "prev_layer", None)
    return chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_gen_warm")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--weight-norm", action="store_true")
    ap.add_argument("--wn-ckpt", default=None)   # defaults to <ckpt>_wn
    a = ap.parse_args()
    img, txt, mask = D.load_batch(2000, seed=0)
    img = np.asarray(img[:a.k], np.float32); txt = np.asarray(txt[:a.k], np.float32); mask = np.asarray(mask[:a.k], np.float32)
    T = tf.convert_to_tensor
    print(f"ckpt={a.ckpt} k={a.k}", flush=True)
    print(f"TRUE image: mean={img.mean():.4f} std={img.std():.4f} min={img.min():.4f} max={img.max():.4f}", flush=True)

    wn_ckpt = a.wn_ckpt or (a.ckpt + "_wn")
    m = build_restore(a.ckpt, T(img), T(txt), T(mask), a.weight_norm, wn_ckpt)

    # Regime I: image-set latent (recon), both clamped, relax
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(T(img), T(txt), T(mask))
    for _ in range(30):
        for L in m.trainable_layers:
            L.update_state()
    li = latent_stats(m)
    print("LATENT (regime I, image-set) norm/std per scale:", [f"{n:.1f}/{s:.3f}" for n, s in li], flush=True)

    # Regime T: text-set latent, caption clamped, image zeros unclamped, relax + top-down boost
    chain = decode_chain(m._infonce_codes[0])
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(tf.zeros_like(T(img)), T(txt), T(mask))
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
    lt = latent_stats(m)
    print("LATENT (regime T, text-set) norm/std per scale:", [f"{n:.1f}/{s:.3f}" for n, s in lt], flush=True)
    print("LATENT norm ratio T/I per scale:", [f"{lt[i][0]/(li[i][0]+1e-9):.2f}" for i in range(len(li))], flush=True)
    gen = m.img_input.predict_next().numpy()
    print(f"GEN image (text-driven, boosted): mean={gen.mean():.4f} std={gen.std():.4f} min={gen.min():.4f} max={gen.max():.4f}", flush=True)
    print(f"  -> gen/true mean ratio={gen.mean()/(img.mean()+1e-9):.3f}, std ratio={gen.std()/(img.std()+1e-9):.3f}", flush=True)
    print("DARKNESS_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
