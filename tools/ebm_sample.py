"""EBM sampling retest: generate by NOISY (Langevin) relaxation from a CD-trained EBM ckpt.
Condition on the caption (text-set latents), then free the image + decode and noisy-relax with an
annealed temperature -> a SAMPLE (latents held fixed). Report brightness/contrast per draw (caveat:
noise inflates std) and save an image grid PNG (true row + one row per draw) -- the visual is the
real judge of whether B1 gives sharp, caption-specific, varying samples. Throwaway."""
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
    ap.add_argument("--noise-temp", type=float, default=100.0)
    ap.add_argument("--relax", type=int, default=60)
    ap.add_argument("--k", type=int, default=6)       # captions (columns)
    ap.add_argument("--draws", type=int, default=3)   # samples per caption (rows)
    ap.add_argument("--out", default="ebm_samples.png")
    a = ap.parse_args()
    wn = a.wn_ckpt or (a.ckpt + "_wn")
    img, txt, mask = D.load_batch(2000, seed=0)
    img = np.asarray(img[:a.k], np.float32); txt = np.asarray(txt[:a.k], np.float32); mask = np.asarray(mask[:a.k], np.float32)
    T = tf.convert_to_tensor
    print(f"ckpt={a.ckpt} noise_temp={a.noise_temp} relax={a.relax} k={a.k} draws={a.draws}", flush=True)
    print(f"TRUE mean={img.mean():.4f} std={img.std():.4f}", flush=True)
    m = build_restore(a.ckpt, T(img), T(txt), T(mask), wn)

    latent_ids = set()
    for a_, b_ in m._shared_latent_pairs:
        latent_ids.add(id(a_)); latent_ids.add(id(b_))
    decode = [L for L in m._image_path_layers
              if hasattr(L, "update_wts") and id(L) not in latent_ids and L is not m.img_input]
    noised = decode + [m.img_input]

    draws = []
    for d in range(a.draws):
        # condition on the caption -> text-set latents (held fixed during sampling)
        m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(tf.zeros_like(T(img)), T(txt), T(mask))
        m.img_input.is_clamped = False
        for _ in range(15):
            for L in m.trainable_layers:
                L.update_state()
            m.img_input.update_state()
        # sample the image + decode by annealed noisy relaxation (latents excluded -> fixed)
        for i in range(a.relax):
            t = a.noise_temp * (1.0 - i / float(a.relax))
            for L in noised:
                L.noise_temp = t
            for L in decode:
                L.update_state()
            m.img_input.update_state()
        for L in noised:
            L.noise_temp = 0.0
        gen = m.img_input.predict_next().numpy()
        draws.append(gen)
        print(f"draw {d}: mean={gen.mean():.4f} std={gen.std():.4f} min={gen.min():.4f} max={gen.max():.4f} "
              f"-> mean_ratio={gen.mean()/(img.mean()+1e-9):.3f} std_ratio={gen.std()/(img.std()+1e-9):.3f}", flush=True)

    rows = 1 + a.draws
    fig, ax = plt.subplots(rows, a.k, figsize=(a.k * 1.6, rows * 1.6))
    for j in range(a.k):
        ax[0][j].imshow(np.clip(img[j], 0, 1)); ax[0][j].axis("off")
        if j == 0: ax[0][j].set_title("true", fontsize=8, loc="left")
    for d in range(a.draws):
        for j in range(a.k):
            ax[d + 1][j].imshow(np.clip(draws[d][j], 0, 1)); ax[d + 1][j].axis("off")
            if j == 0: ax[d + 1][j].set_title(f"sample {d}", fontsize=8, loc="left")
    plt.tight_layout(); plt.savefig(a.out, dpi=90); print(f"saved {a.out}", flush=True)
    print("EBM_SAMPLE_DONE", flush=True)


if __name__ == "__main__":
    main()
