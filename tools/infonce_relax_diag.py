"""Energy-term diagnostic (read-only, fixed weights): can the RELAXATION alone drive
the deepest branch codes (inter2/inter12) into cross-modal alignment if we (a) relax
far longer than 15 substeps and (b) scale the InfoNCE term to the recon-update
magnitude so it stays significant as recon converges?

Decides Approach A vs B for the energy-term design:
- if infonce_loss descends BELOW chance (ln8) as substeps grow -> the states CAN align
  at fixed weights; the training failure was under-relaxation -> Approach A viable.
- if loss stays pinned at chance even at 200 substeps under strong/scaled injection ->
  relaxation cannot align these codes at fixed weights -> need Approach B (direct local
  weight force).

No weight updates, no training. Throwaway. Reuses infonce.infonce_grads.
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
from infonce import infonce_grads

CKPTS = [1, 5, 15, 30, 60, 100, 150, 200]
CLIP = 400.0


def cos_stats(u, v):
    un = tf.math.l2_normalize(u, 1); vn = tf.math.l2_normalize(v, 1)
    S = tf.matmul(un, vn, transpose_b=True)
    B = S.shape[0]
    eye = tf.eye(B)
    matched = float(tf.reduce_sum(S * eye) / B)
    mism = float(tf.reduce_sum(S * (1.0 - eye)) / (B * (B - 1)))
    return matched, mism


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
    latest = tf.train.latest_checkpoint(ckpt)
    assert latest, f"no checkpoint in {ckpt}"
    ck.restore(latest).expect_partial()
    print(f"restored {latest} ({len(ALL_W)} vars)", flush=True)
    return m


def probe(m, img, txt, mask, mode, lam, tau, substeps):
    # fresh forward init from the clamped inputs for each mode
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    ui, vi = m._infonce_codes
    rows = []
    for s in range(1, substeps + 1):
        b2 = tf.identity(ui.state); b12 = tf.identity(vi.state)   # for recon-delta scaling
        for L in m.trainable_layers:
            L.update_state()
        du, dv, _, _ = infonce_grads(ui.state, vi.state, tau)
        if mode == "fixed":
            ui.state.assign_sub(ui.state_lr * lam * du)
            vi.state.assign_sub(vi.state_lr * lam * dv)
        elif mode == "scaled":   # match the alignment step to this substep's recon-update norm
            s2 = tf.norm(ui.state - b2) / (tf.norm(du) + 1e-9)
            s12 = tf.norm(vi.state - b12) / (tf.norm(dv) + 1e-9)
            ui.state.assign_sub(lam * s2 * du)
            vi.state.assign_sub(lam * s12 * dv)
        if mode != "none":       # keep magnitudes bounded (cosine is scale-free; this is just numeric safety)
            ui.state.assign(tf.clip_by_value(ui.state, -CLIP, CLIP))
            vi.state.assign(tf.clip_by_value(vi.state, -CLIP, CLIP))
        if s in CKPTS:
            _, _, acc, loss = infonce_grads(ui.state, vi.state, tau)
            mc, mm = cos_stats(ui.state, vi.state)
            rows.append((s, float(loss), float(acc), mc, mm))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_gelu_best")
    ap.add_argument("--substeps", type=int, default=200)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.07)
    a = ap.parse_args()

    img, txt, mask = D.load_batch(2000, seed=0)
    img = tf.convert_to_tensor(np.asarray(img[:a.k], np.float32))
    txt = tf.convert_to_tensor(np.asarray(txt[:a.k], np.float32))
    mask = tf.convert_to_tensor(np.asarray(mask[:a.k], np.float32))
    print(f"ckpt={a.ckpt} k={a.k} substeps={a.substeps} tau={a.tau} chance_loss=ln({a.k})={np.log(a.k):.4f}", flush=True)

    m = build_restore(a.ckpt, img, txt, mask)

    modes = [("none", 0.0), ("fixed", 0.3), ("fixed", 1.0), ("fixed", 3.0),
             ("scaled", 0.5), ("scaled", 1.0)]
    for mode, lam in modes:
        rows = probe(m, img, txt, mask, mode, lam, a.tau, a.substeps)
        tag = f"{mode}" + (f"(lam={lam})" if mode != "none" else "")
        print(f"\n=== mode={tag} ===", flush=True)
        print(" substep |   loss  |  acc  | cos_matched cos_mism   gap", flush=True)
        for (s, loss, acc, mc, mm) in rows:
            print(f"  {s:4d}   | {loss:6.4f} | {acc:5.3f} |   {mc:+.4f}    {mm:+.4f}  {mc-mm:+.4f}", flush=True)
    print("\nRELAX_DIAG_DONE", flush=True)


if __name__ == "__main__":
    main()
