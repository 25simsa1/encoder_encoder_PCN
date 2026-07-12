"""Reverse-diffusion sampling from a PC-native denoiser. Start from pure noise, clamp the caption,
and walk the noise schedule down: at each level clamp x_t, relax so the encoder maps x_t+caption to
the latent and the decode predicts the clean image x0_hat, then re-noise to the next lower level.
Saves an image grid (true row + one row per draw). Throwaway."""
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
    print(f"restored {tf.train.latest_checkpoint(ckpt)}", flush=True)
    for L in m.trainable_layers:
        if hasattr(L, "enable_weight_norm") and getattr(L, "wts", None) is not None:
            L.enable_weight_norm()
    WN = [L.g_mag for L in m.trainable_layers if getattr(L, "weight_norm", False)]
    wck = tf.train.Checkpoint(**{f"g{i}": v for i, v in enumerate(WN)})
    wck.restore(tf.train.latest_checkpoint(wn_ckpt)).expect_partial()
    print(f"restored wn {tf.train.latest_checkpoint(wn_ckpt)}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wn-ckpt", default=None)
    ap.add_argument("--levels", type=int, default=10)
    ap.add_argument("--sigma-min", type=float, default=0.05)
    ap.add_argument("--sigma-max", type=float, default=0.8)
    ap.add_argument("--k", type=int, default=15)      # encode-relax steps per level
    ap.add_argument("--kcap", type=int, default=6)     # captions (columns)
    ap.add_argument("--draws", type=int, default=3)    # samples per caption (rows)
    ap.add_argument("--out", default="diffusion_samples.png")
    a = ap.parse_args()
    wn = a.wn_ckpt or (a.ckpt + "_wn")
    sigmas = np.geomspace(a.sigma_min, a.sigma_max, a.levels).astype(np.float32)  # ascending
    img, txt, mask = D.load_batch(2000, seed=0)
    img = np.asarray(img[:a.kcap], np.float32); txt = np.asarray(txt[:a.kcap], np.float32); mask = np.asarray(mask[:a.kcap], np.float32)
    T = tf.convert_to_tensor
    print(f"ckpt={a.ckpt} levels={a.levels} sigma=[{a.sigma_min},{a.sigma_max}] k={a.k}", flush=True)
    print(f"TRUE mean={img.mean():.4f} std={img.std():.4f}", flush=True)
    m = build_restore(a.ckpt, T(img), T(txt), T(mask), wn)
    conv1 = [L for L in m._image_path_layers if getattr(L, "prev_layer", None) is m.img_input][0]

    draws = []
    for d in range(a.draws):
        x = (sigmas[-1] * np.random.normal(size=img.shape)).astype(np.float32)   # x_N, pure noise
        for i in reversed(range(a.levels)):                                       # t = N..1
            m.img_input.set_state(T(x)); m.img_input.is_clamped = True; m.txt_input.is_clamped = True
            m.pass_through(T(x), T(txt), T(mask))
            for _ in range(a.k):
                for L in m.trainable_layers:
                    L.update_state()
            x0_hat = conv1.predict_prev().numpy()
            if i > 0:
                x = (x0_hat + sigmas[i - 1] * np.random.normal(size=img.shape)).astype(np.float32)
            else:
                x = x0_hat
        draws.append(x)
        g = x
        print(f"draw {d}: mean={g.mean():.4f} std={g.std():.4f} min={g.min():.4f} max={g.max():.4f} "
              f"-> mean_ratio={g.mean()/(img.mean()+1e-9):.3f} std_ratio={g.std()/(img.std()+1e-9):.3f}", flush=True)

    rows = 1 + a.draws
    fig, ax = plt.subplots(rows, a.kcap, figsize=(a.kcap * 1.6, rows * 1.6))
    for j in range(a.kcap):
        ax[0][j].imshow(np.clip(img[j], 0, 1)); ax[0][j].axis("off")
        if j == 0: ax[0][j].set_title("true", fontsize=8, loc="left")
    for d in range(a.draws):
        for j in range(a.kcap):
            ax[d + 1][j].imshow(np.clip(draws[d][j], 0, 1)); ax[d + 1][j].axis("off")
            if j == 0: ax[d + 1][j].set_title(f"sample {d}", fontsize=8, loc="left")
    plt.tight_layout(); plt.savefig(a.out, dpi=90); print(f"saved {a.out}", flush=True)
    print("DIFFUSION_SAMPLE_DONE", flush=True)


if __name__ == "__main__":
    main()
