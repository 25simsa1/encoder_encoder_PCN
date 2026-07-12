"""Direct denoiser test for the diffusion ckpt: does the net clean a LIGHTLY noised TRUE image?
For each of a few sigmas, x_t = x0 + sigma*eps, encode x_t + caption, free the image and relax the
decode (latent fixed) -> x0_hat. Grid rows: true, then (noised x_t, denoised x0_hat) per sigma. If it
cleans small noise but not pure noise, the denoiser works and the failure is the image-dominated
latent at high-noise steps (caption cannot drive from noise). Throwaway."""
import argparse
import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import COCO64_GEN as C
from conv_pcn_layer import Conv2DPCNLayer
import coco64_data as D
CLIP = 400.0


def build_restore(ckpt, img, txt, mask, wn_ckpt):
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
    for L in m.trainable_layers:
        if hasattr(L, "enable_weight_norm") and getattr(L, "wts", None) is not None:
            L.enable_weight_norm()
    WN = [L.g_mag for L in m.trainable_layers if getattr(L, "weight_norm", False)]
    wck = tf.train.Checkpoint(**{f"g{i}": v for i, v in enumerate(WN)})
    wck.restore(tf.train.latest_checkpoint(wn_ckpt)).expect_partial()
    print(f"restored {tf.train.latest_checkpoint(ckpt)} + wn", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--kcap", type=int, default=6)
    ap.add_argument("--out", default="denoise_test.png")
    a = ap.parse_args()
    wn = a.ckpt + "_wn"
    sigmas = [0.1, 0.3, 0.6]
    img, txt, mask = D.load_batch(2000, seed=0)
    img = np.asarray(img[:a.kcap], np.float32); txt = np.asarray(txt[:a.kcap], np.float32); mask = np.asarray(mask[:a.kcap], np.float32)
    T = tf.convert_to_tensor
    m = build_restore(a.ckpt, T(img), T(txt), T(mask), wn)
    latent_ids = set()
    for a_, b_ in m._shared_latent_pairs:
        latent_ids.add(id(a_)); latent_ids.add(id(b_))
    decode = [L for L in m._image_path_layers
              if hasattr(L, "update_wts") and id(L) not in latent_ids and L is not m.img_input]

    rows = [("true", img)]
    for s in sigmas:
        x_t = (img + s * np.random.normal(size=img.shape)).astype(np.float32)
        m.img_input.set_state(T(x_t)); m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(T(x_t), T(txt), T(mask))
        for _ in range(a.k):
            for L in m.trainable_layers:
                L.update_state()
        m.img_input.is_clamped = False
        for _ in range(a.k):
            for L in decode:
                L.update_state()
            m.img_input.update_state()
        x0_hat = np.clip(m.img_input.predict_next().numpy(), 0.0, 1.0)
        err = float(np.mean((x0_hat - img) ** 2))
        print(f"sigma={s}: denoised MSE-to-true={err:.4f} (noised MSE={float(np.mean((np.clip(x_t,0,1)-img)**2)):.4f})", flush=True)
        rows.append((f"noised s={s}", np.clip(x_t, 0, 1)))
        rows.append((f"denoised s={s}", x0_hat))

    fig, ax = plt.subplots(len(rows), a.kcap, figsize=(a.kcap * 1.6, len(rows) * 1.6))
    for r, (lab, im) in enumerate(rows):
        for j in range(a.kcap):
            ax[r][j].imshow(np.clip(im[j], 0, 1)); ax[r][j].axis("off")
            if j == 0: ax[r][j].set_title(lab, fontsize=8, loc="left")
    plt.tight_layout(); plt.savefig(a.out, dpi=90); print(f"saved {a.out}", flush=True)
    print("DENOISE_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
