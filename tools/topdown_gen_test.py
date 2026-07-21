"""DECISIVE test (read-only, fixed weights): can a top-down-authoritative relaxation
generate a DIVERSE, structured image from a caption at INFERENCE, with no retraining?

Drive-balance was confirmed: the decode weights carry per-caption diversity but the
symmetric PC relaxation lets the uninformed bottom-up (zero image) collapse it at every
layer. Here we boost the top-down pull along the WHOLE image-decode chain (inter2 ->
dense1 -> ... -> conv1 -> img_input, via prev_layer walk) during text->image generation,
leaving the shared latent FREE so text still sets it. Sweep the boost gamma; measure the
generated-image participation ratio (PR: 1=blob, up to K=diverse) and save PNGs.

If the image PR rises off ~1 and the PNGs show caption-varying structure -> text->image
works at inference, free fix. If it stays collapsed -> retraining is needed. Throwaway.
"""
import argparse, os
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
CLIP = 400.0
OUT = "gen_td_out"


def pr(states):
    X = tf.reshape(states, (states.shape[0], -1)); X = X - tf.reduce_mean(X, 0, keepdims=True)
    v = tf.linalg.svd(X, compute_uv=False) ** 2
    return float((tf.reduce_sum(v) ** 2) / (tf.reduce_sum(v ** 2) + 1e-12))


def to_png(arr, path):
    a = np.clip(np.asarray(arr, np.float32), 0, 1)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(path)
    return os.path.getsize(path)


def build_restore(ckpt, img, txt, mask):
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
    latest = tf.train.latest_checkpoint(ckpt); assert latest
    ck.restore(latest).expect_partial(); print(f"restored {latest}", flush=True)
    return m


def decode_chain(ui):
    chain, L = [], ui
    while L is not None:
        chain.append(L); L = getattr(L, "prev_layer", None)
    return chain   # inter2 -> dense1 -> ... -> conv1 -> img_input


def gen(m, chain, zeros_img, txt, mask, gamma, steps):
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(zeros_img, txt, mask)
    m.img_input.is_clamped = False; m.txt_input.is_clamped = True
    for _ in range(steps):
        for L in m.trainable_layers:
            L.update_state()
        m.img_input.update_state()
        if gamma > 0:      # boost top-down authority along the decode chain (latent left free)
            for L in chain:
                st = getattr(L, "state", None)
                if st is None or not L.next_layers:
                    continue
                nxt = L.next_layers[0]
                if not hasattr(nxt, "predict_prev"):   # e.g. maxpool above; skip
                    continue
                td = nxt.predict_prev()
                if td.shape != st.shape:
                    continue
                st.assign(tf.clip_by_value(st + gamma * (td - st), -CLIP, CLIP))
    return m.img_input.predict_next()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_gen_best")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    img, txt, mask = D.load_batch(2000, seed=0)
    T = tf.convert_to_tensor
    txt = T(np.asarray(txt[:a.k], np.float32)); mask = T(np.asarray(mask[:a.k], np.float32))
    caps = open(f"{D.CACHE}/caps_sc_train2017.txt").read().splitlines()[:a.k]
    zeros_img = tf.zeros((a.k, C.img_resolution, C.img_resolution, 3), tf.float32)
    print(f"ckpt={a.ckpt} k={a.k} relax={RELAX} (image PR: 1=blob, up to {a.k}=diverse)", flush=True)

    m = build_restore(a.ckpt, zeros_img, txt, mask)
    chain = decode_chain(m._infonce_codes[0])
    print(f"decode chain length={len(chain)} ({type(chain[0]).__name__}..{type(chain[-1]).__name__})", flush=True)

    print("\n gamma | image PR | t2i png bytes (min/mean/max)", flush=True)
    for gamma in [0.0, 0.5, 1.0, 2.0]:
        out = gen(m, chain, zeros_img, txt, mask, gamma, RELAX).numpy()
        szs = []
        for i in range(min(a.k, 4)):
            szs.append(to_png(out[i], f"{OUT}/g{gamma}_{i}.png"))
        print(f"  {gamma:4.1f} |  {pr(out):6.2f}  | {min(szs)}/{int(np.mean(szs))}/{max(szs)}", flush=True)
    open(f"{OUT}/caps.txt", "w").write("\n".join(caps[:4]))
    print("\nTOPDOWN_GEN_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
