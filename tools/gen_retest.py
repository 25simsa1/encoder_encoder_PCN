"""ID.5 deliverable: text->image retest on the COCO64_GEN checkpoint (invertible
strided-conv downsampling). With the generative pathway reconnected through the
downsamplers, the model's OWN test_step relaxation should now propagate top-down to
the pixels. KEY question: does text->image produce caption-VARYING STRUCTURE (not the
uniform blob), with reconstruction + image->caption intact?

Uses the model's standard test_step (no manual top-down boost). Fixed batch K across
all test_step calls (one instance cannot switch batch size). Throwaway.
"""
import argparse, json, os
import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import COCO64_GEN as C
from conv_pcn_layer import Conv2DPCNLayer
import coco64_data as D
from PIL import Image

RELAX = 150
OUT = "gen_retest_out"


def to_png(arr, path):
    a = np.clip(np.asarray(arr, np.float32), 0.0, 1.0)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(path)
    return os.path.getsize(path)


def pr(states):
    X = tf.reshape(tf.convert_to_tensor(states), (states.shape[0], -1))
    X = X - tf.reduce_mean(X, axis=0, keepdims=True)
    v = tf.linalg.svd(X, compute_uv=False) ** 2
    return float((tf.reduce_sum(v) ** 2) / (tf.reduce_sum(v ** 2) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_gen_best")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--relax", type=int, default=RELAX)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    img, txt, mask = D.load_batch(2000, seed=0)
    img = np.asarray(img[:a.k], np.float32); txt = np.asarray(txt[:a.k], np.float32); mask = np.asarray(mask[:a.k], np.float32)
    caps = open(f"{D.CACHE}/caps_sc_train2017.txt").read().splitlines()[:a.k]
    print(f"ckpt={a.ckpt} k={a.k} relax={a.relax} (image PR: 1=blob, up to {a.k}=diverse)", flush=True)

    m = EncoderEncoderPCN(1e-4, config=C)
    for L in m.trainable_layers:
        if hasattr(L, "state_clip"):
            L.state_clip = 400.0
    for L in m.trainable_layers:                 # gelu on stride-1 convs only (downsamplers stay linear, matching training)
        if isinstance(L, Conv2DPCNLayer) and getattr(L, "stride", 1) == 1:
            L.activation = "gelu"
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(tf.convert_to_tensor(img), tf.convert_to_tensor(txt), tf.convert_to_tensor(mask))
    ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts", "b", "gamma", "beta")
             if isinstance(getattr(l, at, None), tf.Variable)]
    ck = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    ckdir = a.ckpt if tf.train.latest_checkpoint(a.ckpt) else "ckpt_gen"
    latest = tf.train.latest_checkpoint(ckdir); assert latest, f"no checkpoint in {a.ckpt} or ckpt_gen"
    ck.restore(latest).expect_partial(); print(f"restored {latest} ({len(ALL_W)} vars)", flush=True)

    T = tf.convert_to_tensor
    zeros_img = tf.zeros_like(T(img)); zeros_txt = tf.zeros_like(T(txt))
    t2i = m.test_step(a.relax, zeros_img, T(txt), predict='img', mask=T(mask)).numpy()   # caption clamped, image from zero init
    recon = m.test_step(a.relax, T(img), T(txt), predict='img', mask=T(mask)).numpy()    # real image init (recon-leaning)
    i2t = m.test_step(a.relax, T(img), zeros_txt, predict='txt', mask=T(mask)).numpy()   # image clamped -> caption

    rows = []
    for i in range(a.k):
        s_t2i = to_png(t2i[i], f"{OUT}/t2i_{i}.png"); s_rec = to_png(recon[i], f"{OUT}/recon_{i}.png"); s_tru = to_png(img[i], f"{OUT}/true_{i}.png")
        dec = D.decode(i2t[i])
        st = float(np.asarray(t2i[i]).std())
        rows.append(dict(i=i, t2i_png=s_t2i, t2i_std=round(st, 5), recon_png=s_rec, true_png=s_tru,
                         true_cap=caps[i][:50], decoded_cap=dec[:50]))
        print(f"[{i}] t2i_png={s_t2i}B std={st:.5f} | recon_png={s_rec}B | cap_true='{caps[i][:42]}' -> decoded='{dec[:42]}'", flush=True)

    print(f"\nSUMMARY text->image: image_PR={pr(t2i):.3f} (1=blob, up to {a.k}); png_bytes min/mean/max={min(r['t2i_png'] for r in rows)}/{int(np.mean([r['t2i_png'] for r in rows]))}/{max(r['t2i_png'] for r in rows)}; std mean={np.mean([r['t2i_std'] for r in rows]):.5f}", flush=True)
    print(f"reconstruction PR={pr(recon):.3f}", flush=True)
    print("(STRUCTURE => image_PR well above 1 AND per-image PNGs differ/show content; blob => PR~1, ~91B, std~0)", flush=True)
    json.dump(rows, open(f"{OUT}/results.json", "w"), indent=2)
    print("GEN_RETEST_DONE", flush=True)


if __name__ == "__main__":
    main()
