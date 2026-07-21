"""Fast lambda-response probe for the scale-robust InfoNCE coupling nudge. Load an encoder,
run the coupled relaxation on ONE fixed batch across a lambda grid, and report how far the
text code moves and whether the batch InfoNCE accuracy rises above chance (1/B). No weight
updates, no training. Answers 'which lambda actually bites' in ~1 minute. Throwaway."""
import argparse
import numpy as np
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
import os
from pcn_config import COCO64_GEN, COCO64_WIDE
C = COCO64_WIDE if os.environ.get('PCN_TOOL_CONFIG') == 'coco64_wide' else COCO64_GEN
from conv_pcn_layer import Conv2DPCNLayer
from infonce import infonce_grads
import coco64_data as D
CLIP = 400.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_wide5_best")
    ap.add_argument("--batch", type=int, default=32)   # bigger batch = harder, more negatives
    ap.add_argument("--relax", type=int, default=15)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--anchor-img", type=int, default=1)
    a = ap.parse_args()
    img, txt, mask = D.load_batch(a.batch, seed=0)
    T = tf.convert_to_tensor
    m = EncoderEncoderPCN(1e-3, config=C)
    for L in m.trainable_layers:
        if hasattr(L, "state_clip"):
            L.state_clip = CLIP
        if isinstance(L, Conv2DPCNLayer) and getattr(L, "stride", 1) == 1:
            L.activation = "gelu"
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(T(np.asarray(img, np.float32)), T(np.asarray(txt, np.float32)),
                   T(np.asarray(mask, np.float32)))
    ALL_W = [getattr(l, at) for l in m.trainable_layers for at in ("wts", "b", "gamma", "beta")
             if isinstance(getattr(l, at, None), tf.Variable)]
    ck = tf.train.Checkpoint(**{f"v{i}": v for i, v in enumerate(ALL_W)})
    ck.restore(tf.train.latest_checkpoint(a.ckpt)).expect_partial()
    print(f"restored {a.ckpt}; batch={a.batch} chance_acc={1.0/a.batch:.3f}", flush=True)
    ui, vi = m._infonce_codes

    def nudge(L, d, lam):
        sc = tf.sqrt(tf.reduce_mean(tf.square(L.state))) + 1e-6
        L.state.assign_sub(lam * sc * d / (tf.norm(d) + 1e-9))

    for lam in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5]:
        m.img_input.is_clamped = True; m.txt_input.is_clamped = True
        m.pass_through(T(np.asarray(img, np.float32)), T(np.asarray(txt, np.float32)),
                       T(np.asarray(mask, np.float32)))
        v0 = tf.identity(vi.state)
        acc = loss = None
        for _ in range(a.relax):
            for L in m.trainable_layers:
                L.update_state()
            du, dv, acc, loss = infonce_grads(ui.state, vi.state, a.tau)
            if lam > 0:
                if not a.anchor_img:
                    nudge(ui, du, lam)
                nudge(vi, dv, lam)
        drift = float(tf.norm(vi.state - v0) / (tf.norm(v0) + 1e-9))
        print(f"lam={lam:4.2f}  acc={float(acc):.3f}  loss={float(loss):.4f}  "
              f"text_code_drift={drift:.3f}", flush=True)
    print("NCE_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
