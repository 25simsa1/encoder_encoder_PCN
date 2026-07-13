"""Latent-source decode probe (read-only, no training, pure bidirectional PC). Does the
standalone top-down decode produce a SHARP image when the latents hold the RIGHT content
(set by the image itself), versus the known blur when set by the text?

- Capture IMAGE-set latents (recon clamp config, relax) and TEXT-set latents (caption clamped,
  image zeros unclamped, boosted relax, the established generation regime).
- Decode protocol, identical for every latent source: assign the captured latents, hold them
  fixed (unclamped, excluded from every loop), free the image, relax the decode with the
  top-down boost, read the image.
- Per-scale swaps: text-set latents with ONE scale replaced by its image-set version, to
  localize which scale carries the missing detail.

Sharp image-set decode + blurry text-set decode => the latent CONTENT is the bottleneck (fix
the text drive into the fine latents). Blurry both => the standalone decode itself is the
ceiling and no latent fix will crack it. Throwaway.
"""
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
GAMMA = 1.0   # overridden by --gamma; 0 = plain relaxation (no boost)


def build_restore(ckpt, img, txt, mask, weight_norm, wn_ckpt):
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
    if weight_norm:
        for L in m.trainable_layers:
            if hasattr(L, "enable_weight_norm") and getattr(L, "wts", None) is not None:
                L.enable_weight_norm()
        WN = [L.g_mag for L in m.trainable_layers if getattr(L, "weight_norm", False)]
        wck = tf.train.Checkpoint(**{f"g{i}": v for i, v in enumerate(WN)})
        wck.restore(tf.train.latest_checkpoint(wn_ckpt)).expect_partial()
        print(f"restored wn {tf.train.latest_checkpoint(wn_ckpt)}", flush=True)
    return m


def decode_chain(ui):
    chain, L = [], ui
    while L is not None:
        chain.append(L); L = getattr(L, "prev_layer", None)
    return chain


def boost(chain, latent_ids):
    # top-down boost along the decode chain (the established generation schedule); never
    # moves the fixed latents; GAMMA==0 disables it (plain relaxation)
    if GAMMA == 0.0:
        return
    for L in chain:
        st = getattr(L, "state", None)
        if st is None or not L.next_layers or id(L) in latent_ids:
            continue
        nxt = L.next_layers[0]
        if not hasattr(nxt, "predict_prev"):
            continue
        td = nxt.predict_prev()
        if td.shape != st.shape:
            continue
        st.assign(tf.clip_by_value(st + GAMMA * (td - st), -CLIP, CLIP))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_gen_best")
    ap.add_argument("--weight-norm", action="store_true")
    ap.add_argument("--wn-ckpt", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--relax-cap", type=int, default=150)   # capture/decode relax steps
    ap.add_argument("--gamma", type=float, default=1.0)     # boost strength; 0 = plain relaxation
    ap.add_argument("--pi-bu", type=float, default=None)    # generative precision schedule for the decode; None = off
    ap.add_argument("--out", default="latent_source.png")
    a = ap.parse_args()
    global GAMMA
    GAMMA = a.gamma
    img, txt, mask = D.load_batch(2000, seed=0)
    img = np.asarray(img[:a.k], np.float32); txt = np.asarray(txt[:a.k], np.float32); mask = np.asarray(mask[:a.k], np.float32)
    T = tf.convert_to_tensor
    print(f"ckpt={a.ckpt} k={a.k} relax={a.relax_cap} gamma={GAMMA}", flush=True)
    print(f"TRUE mean={img.mean():.4f} std={img.std():.4f}", flush=True)
    m = build_restore(a.ckpt, T(img), T(txt), T(mask), a.weight_norm, a.wn_ckpt or (a.ckpt + "_wn"))

    pairs = m._shared_latent_pairs
    latent_ids = set()
    for a_, b_ in pairs:
        latent_ids.add(id(a_)); latent_ids.add(id(b_))
    decode = [L for L in m._image_path_layers
              if hasattr(L, "update_wts") and id(L) not in latent_ids and L is not m.img_input]
    chain = decode_chain(m._infonce_codes[0])

    # --- capture IMAGE-set latents (recon clamp config, relax; the correct-content latents)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(T(img), T(txt), T(mask))
    for _ in range(30):
        for L in m.trainable_layers:
            L.update_state()
    lat_img = [a_.state.numpy().copy() for a_, _ in pairs]

    # --- capture TEXT-set latents (caption clamped, image zeros unclamped, boosted relax;
    #     the established generation regime)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(tf.zeros_like(T(img)), T(txt), T(mask))
    m.img_input.is_clamped = False
    for _ in range(a.relax_cap):
        for L in m.trainable_layers:
            L.update_state()
        m.img_input.update_state()
        boost(chain, set())     # capture regime boosts everything, as darkness_diag does
    lat_txt = [a_.state.numpy().copy() for a_, _ in pairs]
    for i in range(len(pairs)):
        ni = float(np.linalg.norm(lat_img[i])); nt = float(np.linalg.norm(lat_txt[i]))
        print(f"latent scale {i}: |img|={ni:.1f} |txt|={nt:.1f} T/I={nt/(ni+1e-9):.2f}", flush=True)

    def decode_from(latents, label):
        # identical standalone decode for every latent source: latents fixed, image free
        m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(tf.zeros_like(T(img)), T(txt), T(mask))
        for (a_, _), val in zip(pairs, latents):
            a_.state.assign(val)
        m.img_input.is_clamped = False
        if a.pi_bu is not None:
            for L in decode:
                L.pi_bu = a.pi_bu   # generative precision schedule: top-down-dominant relaxation
        for _ in range(a.relax_cap):
            for L in decode:
                L.update_state()
            m.img_input.update_state()
            boost(chain, latent_ids)    # boost the decode chain; never the fixed latents
        if a.pi_bu is not None:
            for L in decode:
                L.pi_bu = 1.0       # restore the default so the next capture phase is standard
        gen = m.img_input.predict_next().numpy()
        mse = float(np.mean((np.clip(gen, 0, 1) - img) ** 2))
        print(f"{label}: mean={gen.mean():.4f} std={gen.std():.4f} "
              f"mean_ratio={gen.mean()/(img.mean()+1e-9):.3f} std_ratio={gen.std()/(img.std()+1e-9):.3f} "
              f"mse_to_true={mse:.4f}", flush=True)
        return gen

    rows = [("true", img)]
    rows.append(("decode(img-set)", decode_from(lat_img, "DECODE image-set latents")))
    rows.append(("decode(txt-set)", decode_from(lat_txt, "DECODE text-set latents")))
    for s in range(len(pairs)):
        sw = list(lat_txt); sw[s] = lat_img[s]
        rows.append((f"swap s{s}", decode_from(sw, f"DECODE swap scale {s} (img at s{s}, txt elsewhere)")))

    fig, ax = plt.subplots(len(rows), a.k, figsize=(a.k * 1.6, len(rows) * 1.6))
    for r, (lab, im) in enumerate(rows):
        for j in range(a.k):
            ax[r][j].imshow(np.clip(im[j], 0, 1)); ax[r][j].axis("off")
            if j == 0: ax[r][j].set_title(lab, fontsize=8, loc="left")
    plt.tight_layout(); plt.savefig(a.out, dpi=90); print(f"saved {a.out}", flush=True)
    print("LATENT_SOURCE_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
