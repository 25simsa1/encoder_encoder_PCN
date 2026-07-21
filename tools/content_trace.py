"""Content-tracing probe: WHERE does latent content die on the way to the image?
Relax the decode twice (image-set vs text-set latents, identical protocol) and print the
per-layer norm of the state DIFFERENCE between the two runs, for every decode layer plus the
image. The first layer whose diff collapses to ~0 is where the flow is severed. Throwaway."""
import argparse
import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
import os
from pcn_config import COCO64_GEN, COCO64_WIDE
# PCN_TOOL_CONFIG=coco64_wide selects the wide-inter config; default unchanged
C = COCO64_WIDE if os.environ.get('PCN_TOOL_CONFIG') == 'coco64_wide' else COCO64_GEN
from conv_pcn_layer import Conv2DPCNLayer
import coco64_data as D
CLIP = 400.0


def build_restore(ckpt, img, txt, mask, td_ckpt, td_affine):
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
    print(f"restored {ckpt} + td", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_untied5")
    ap.add_argument("--td-affine", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--slr", type=float, default=0.25)
    a = ap.parse_args()
    img, txt, mask = D.load_batch(2000, seed=0)
    img = np.asarray(img[:a.k], np.float32); txt = np.asarray(txt[:a.k], np.float32); mask = np.asarray(mask[:a.k], np.float32)
    T = tf.convert_to_tensor
    m = build_restore(a.ckpt, T(img), T(txt), T(mask), a.ckpt + "_td", a.td_affine)
    pairs = m._shared_latent_pairs
    latent_ids = set()
    for a_, b_ in pairs:
        latent_ids.add(id(a_)); latent_ids.add(id(b_))
    decode = [L for L in m._image_path_layers
              if hasattr(L, "update_wts") and id(L) not in latent_ids and L is not m.img_input]
    names = {}
    for i, L in enumerate(m._image_path_layers):
        names[id(L)] = f"{i:02d}:{type(L).__name__}"

    # capture image-set latents (recon relax) and text-set latents (zeros image, plain relax)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(T(img), T(txt), T(mask))
    for _ in range(30):
        for L in m.trainable_layers:
            L.update_state()
    lat_img = [tf.identity(a_.state) for a_, _ in pairs]
    fwd_rms = {id(L): float(tf.sqrt(tf.reduce_mean(tf.square(L.state)))) for L in decode if getattr(L, "state", None) is not None}
    m.pass_through(tf.zeros_like(T(img)), T(txt), T(mask))
    m.img_input.is_clamped = False
    for _ in range(30):
        for L in m.trainable_layers:
            L.update_state()
        m.img_input.update_state()
    lat_txt = [tf.identity(a_.state) for a_, _ in pairs]
    for i in range(len(pairs)):
        d = float(tf.norm(lat_img[i] - lat_txt[i]))
        print(f"latent s{i}: |img-txt| = {d:.2f}  (|img|={float(tf.norm(lat_img[i])):.2f})", flush=True)

    def run(latents):
        m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(tf.zeros_like(T(img)), T(txt), T(mask))
        for (a_, _), val in zip(pairs, latents):
            a_.state.assign(val)
        m.img_input.is_clamped = False
        orig = [(L, L.pi_bu, L.state_lr) for L in decode]
        for L in decode:
            L.pi_bu = 0.0; L.state_lr = a.slr
        islr = m.img_input.state_lr; m.img_input.state_lr = a.slr
        for _ in range(a.steps):
            for L in decode:
                L.update_state()
            m.img_input.update_state()
            for L in decode:
                st = getattr(L, "state", None)
                if st is None or id(L) not in fwd_rms:
                    continue
                cur = tf.sqrt(tf.reduce_mean(tf.square(st))) + 1e-8
                st.assign(st * (fwd_rms[id(L)] / cur))
        for L, pb, sl in orig:
            L.pi_bu = pb; L.state_lr = sl
        m.img_input.state_lr = islr
        out = {id(L): tf.identity(L.state) for L in decode if getattr(L, "state", None) is not None}
        out["img"] = tf.identity(m.img_input.state)
        return out

    A = run(lat_img)
    B = run(lat_txt)
    print("=== per-layer |state_imgset - state_txtset| after the decode (0 = content severed) ===", flush=True)
    for L in decode:
        if id(L) in A and id(L) in B:
            d = float(tf.norm(A[id(L)] - B[id(L)]))
            n = float(tf.norm(A[id(L)])) + 1e-9
            print(f"{names[id(L)]:>28s}  diff={d:10.4f}  rel={d/n:8.5f}", flush=True)
    d = float(tf.norm(A["img"] - B["img"]))
    print(f"{'IMAGE':>28s}  diff={d:10.4f}", flush=True)
    print("CONTENT_TRACE_DONE", flush=True)


if __name__ == "__main__":
    main()
