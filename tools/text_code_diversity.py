"""Diagnostic (read-only, fixed weights): is the TEXT PATH itself collapsed, or does
the caption produce per-pair-distinct codes that just can't move the shared latent?

Across K distinct in-sample pairs, relax (150 substeps, generation regime) and measure
the per-input distinctness of three states:
  - inter12 : text-branch code (fed by the caption)
  - inter2  : image-branch code
  - dense2  : the SHARED latent (aliased dense2==dense4) both codes feed
under two regimes:
  - T: text clamped (real captions), image generated from zeros  -> does the CAPTION
       produce distinct inter12, and does it move dense2?
  - I: image clamped (real images), text generated               -> baseline: the
       working direction; image should give distinct inter2 and drive dense2.

Distinctness metrics per state (across the K batch items):
  - offdiag_cos : mean pairwise cosine of distinct items. ~1 => COLLAPSED (all same),
                  ~0 => diverse/orthogonal.
  - PR          : participation ratio of the centered codes' variance
                  (Sum s_i^2)^2 / Sum s_i^4. 1 => all variance in one direction
                  (collapse), up to K => fully spread.

Reading: if inter12 is collapsed under T (offdiag_cos~1 / PR~1), the text path itself
cannot distinguish captions -> no alignment method can help; text path is the bottleneck.
If inter12 is diverse under T but dense2 collapses under T (while dense2 is diverse
under I), the text produces distinct codes but cannot move the image-dominated shared
latent -> shared-latent domination is the bottleneck. Throwaway.
"""
import argparse
import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import COCO64_156M as C
from conv_pcn_layer import Conv2DPCNLayer
import coco64_data as D

RELAX = 150


def offdiag_cos(states):
    n = tf.math.l2_normalize(tf.reshape(states, (states.shape[0], -1)), axis=1)
    S = tf.matmul(n, n, transpose_b=True)
    K = S.shape[0]
    eye = tf.eye(K)
    return float(tf.reduce_sum(S * (1.0 - eye)) / (K * (K - 1)))


def participation_ratio(states):
    X = tf.reshape(states, (states.shape[0], -1))
    X = X - tf.reduce_mean(X, axis=0, keepdims=True)
    s = tf.linalg.svd(X, compute_uv=False)
    v = s ** 2
    return float((tf.reduce_sum(v) ** 2) / (tf.reduce_sum(v ** 2) + 1e-12))


def build_restore(ckpt, img, txt, mask):
    m = EncoderEncoderPCN(1e-4, config=C)
    for L in m.trainable_layers:
        if hasattr(L, "state_clip"):
            L.state_clip = 400.0
        if isinstance(L, Conv2DPCNLayer):
            L.activation = "gelu"
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts", "b", "gamma", "beta")
             if isinstance(getattr(l, at, None), tf.Variable)]
    ck = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    latest = tf.train.latest_checkpoint(ckpt)
    assert latest, f"no checkpoint in {ckpt}"
    ck.restore(latest).expect_partial()
    print(f"restored {latest} ({len(ALL_W)} vars)", flush=True)
    return m


def relax_regime(m, img, txt, mask, clamp_img):
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    m.img_input.is_clamped = clamp_img
    m.txt_input.is_clamped = not clamp_img
    for _ in range(RELAX):
        for L in m.trainable_layers:
            L.update_state()
        (m.img_input if not clamp_img else m.txt_input).update_state()   # generation double-update
    ui, vi = m._infonce_codes           # inter2 (img code), inter12 (txt code)
    dense2 = ui.next_layers[0]          # the shared latent both feed (aliased dense2==dense4)
    return ui.state, vi.state, dense2.state


def report(name, inter2, inter12, dense2):
    print(f"\n=== regime {name} ===", flush=True)
    for tag, st in [("inter2 (img code)", inter2), ("inter12(txt code)", inter12), ("dense2 (SHARED)", dense2)]:
        print(f"  {tag}: offdiag_cos={offdiag_cos(st):+.4f}  PR={participation_ratio(st):5.2f}  (K={st.shape[0]})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_gelu_best")
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()

    img, txt, mask = D.load_batch(2000, seed=0)
    img = np.asarray(img[:a.k], np.float32)
    txt = np.asarray(txt[:a.k], np.float32)
    mask = np.asarray(mask[:a.k], np.float32)
    T = tf.convert_to_tensor
    zeros_img = tf.zeros_like(T(img)); zeros_txt = tf.zeros_like(T(txt))
    full_mask = tf.zeros_like(T(mask))   # all-valid, for the text-generated regime
    print(f"ckpt={a.ckpt} k={a.k} relax={RELAX}  (offdiag_cos~1=collapsed, PR~1=collapsed / higher=diverse; max PR={a.k})", flush=True)

    m = build_restore(a.ckpt, T(img), T(txt), T(mask))

    # Regime T: caption clamped (real), image generated from zeros
    i2, i12, d2 = relax_regime(m, zeros_img, T(txt), T(mask), clamp_img=False)
    report("T: text-clamped / image-generated", i2, i12, d2)

    # Regime I: image clamped (real), text generated (baseline: working direction)
    i2, i12, d2 = relax_regime(m, T(img), zeros_txt, full_mask, clamp_img=True)
    report("I: image-clamped / text-generated", i2, i12, d2)

    print("\nTEXT_CODE_DIVERSITY_DONE", flush=True)


if __name__ == "__main__":
    main()
