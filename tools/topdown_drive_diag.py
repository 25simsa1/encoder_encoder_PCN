"""Diagnostic (read-only, fixed weights): is inter2's collapse under text->image a
DRIVE-BALANCE problem (bottom-up from the zero image overrides the top-down latent) or
a LEARNED-WEIGHTS problem (the top-down decode itself maps the diverse latent to a
collapsed inter2)?

Phase A: text->image relax (text clamped, image generated), capture the text-set shared
latent (dense2) and the collapsed relaxed inter2.
Key readout: PR of dense2.predict_prev() = the top-down decode of the diverse latent
into inter2-space (one matmul through dense2.wts^T).
  - PR high  => the decode preserves diversity; inter2 collapses only because bottom-up
                (zero-image) overrides it during relaxation => DRIVE-BALANCE (fixable by
                giving the latent more authority / a generative schedule).
  - PR ~1    => the decode weights collapse the diverse latent => LEARNED-WEIGHTS.

Phase B (confirmation): freeze the diverse shared latent (clamp dense2+dense4), relax
with an extra top-down pull on inter2 (gamma sweep), and watch inter2 + generated-image
PR. If boosting top-down authority un-collapses inter2/image => drive-balance. Throwaway.
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
CLIP = 400.0


def pr(states):
    X = tf.reshape(states, (states.shape[0], -1))
    X = X - tf.reduce_mean(X, axis=0, keepdims=True)
    s = tf.linalg.svd(X, compute_uv=False)
    v = s ** 2
    return float((tf.reduce_sum(v) ** 2) / (tf.reduce_sum(v ** 2) + 1e-12))


def odc(states):
    n = tf.math.l2_normalize(tf.reshape(states, (states.shape[0], -1)), axis=1)
    S = tf.matmul(n, n, transpose_b=True); K = S.shape[0]
    return float(tf.reduce_sum(S * (1.0 - tf.eye(K))) / (K * (K - 1)))


def build_restore(ckpt, img, txt, mask):
    m = EncoderEncoderPCN(1e-4, config=C)
    for L in m.trainable_layers:
        if hasattr(L, "state_clip"):
            L.state_clip = CLIP
        if isinstance(L, Conv2DPCNLayer):
            L.activation = "gelu"
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts", "b", "gamma", "beta")
             if isinstance(getattr(l, at, None), tf.Variable)]
    ck = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    latest = tf.train.latest_checkpoint(ckpt); assert latest, f"no ckpt in {ckpt}"
    ck.restore(latest).expect_partial()
    print(f"restored {latest}", flush=True)
    return m


def text2img_relax(m, zeros_img, txt, mask, steps):
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(zeros_img, txt, mask)
    m.img_input.is_clamped = False; m.txt_input.is_clamped = True
    for _ in range(steps):
        for L in m.trainable_layers:
            L.update_state()
        m.img_input.update_state()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_gelu_best")
    ap.add_argument("--k", type=int, default=12)
    a = ap.parse_args()
    img, txt, mask = D.load_batch(2000, seed=0)
    T = tf.convert_to_tensor
    txt = T(np.asarray(txt[:a.k], np.float32)); mask = T(np.asarray(mask[:a.k], np.float32))
    zeros_img = tf.zeros((a.k, C.img_resolution, C.img_resolution, 3), tf.float32)
    print(f"ckpt={a.ckpt} k={a.k} (PR: 1=collapsed, up to {a.k}=diverse)", flush=True)

    m = build_restore(a.ckpt, zeros_img, txt, mask)
    ui, vi = m._infonce_codes           # inter2 (img code), inter12 (txt code)
    dense2 = ui.next_layers[0]; dense4 = vi.next_layers[0]   # shared latent (aliased)

    # Phase A
    text2img_relax(m, zeros_img, txt, mask, RELAX)
    td_target = dense2.predict_prev()   # top-down decode of the diverse latent -> inter2 space
    print("\n=== Phase A: text-clamped / image-generated ===", flush=True)
    print(f"  dense2 (shared latent) : PR={pr(dense2.state):5.2f} odc={odc(dense2.state):+.4f}", flush=True)
    print(f"  dense2.predict_prev()  : PR={pr(td_target):5.2f} odc={odc(td_target):+.4f}   <-- decode of latent -> inter2", flush=True)
    print(f"  inter2 (relaxed)       : PR={pr(ui.state):5.2f} odc={odc(ui.state):+.4f}   <-- collapsed target", flush=True)

    # snapshot states for restore between gammas
    snap = {id(L): tf.identity(L.state) for L in m.trainable_layers if getattr(L, "state", None) is not None}

    print("\n=== Phase B: freeze diverse latent (clamp dense2+dense4), boost top-down pull on inter2 ===", flush=True)
    print(" gamma | inter2 PR | image PR", flush=True)
    for gamma in [0.0, 0.5, 2.0, 10.0]:
        for L in m.trainable_layers:
            if getattr(L, "state", None) is not None and id(L) in snap:
                L.state.assign(snap[id(L)])
        dense2.is_clamped = True; dense4.is_clamped = True
        m.img_input.is_clamped = False; m.txt_input.is_clamped = True
        for _ in range(80):
            for L in m.trainable_layers:
                L.update_state()
            m.img_input.update_state()
            if gamma > 0:                      # extra top-down authority: pull inter2 toward the frozen latent's decode
                ui.state.assign(tf.clip_by_value(ui.state + gamma * (dense2.predict_prev() - ui.state), -CLIP, CLIP))
        print(f"  {gamma:4.1f} |  {pr(ui.state):6.2f}   |  {pr(m.img_input.state):6.2f}", flush=True)
        dense2.is_clamped = False; dense4.is_clamped = False
    print("\nTOPDOWN_DRIVE_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
