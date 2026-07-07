import os, time, math, argparse
import tensorflow as tf, numpy as np
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN

def golden_state_signature(model):
    sig = {}
    for i, L in enumerate(model.trainable_layers):
        s = getattr(L, "state", None)
        if s is not None:
            sig[i] = float(tf.norm(tf.cast(s, tf.float32)))
    return sig

def run_reference(steps=2, batch=1, seed=0):
    tf.random.set_seed(seed)
    m = EncoderEncoderPCN(1e-4)
    img = tf.random.normal((batch, 572, 572, 3), seed=seed)
    txt = tf.random.normal((batch, 192, 512), seed=seed)
    mask = tf.zeros((batch, 192), tf.float32)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    try: tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception: pass
    t = time.time()
    m.update_states_wts_b(steps)
    dt = (time.time() - t) / max(1, steps)
    peak = 0.0
    try: peak = tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**30
    except Exception: pass
    return golden_state_signature(m), peak, dt

def compare(a, b, tol=1e-4):
    bad = []
    for k in a:
        av = a[k]
        bv = b.get(k, 0.0)
        d = abs(av-bv) / (abs(av) + 1e-9)
        if (not math.isfinite(av)) or (not math.isfinite(bv)) or (d > tol):
            bad.append((k, av, bv, d))
    return bad

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--save", default="golden.npz")
    a = ap.parse_args()
    sig, peak, dt = run_reference(a.steps, a.batch)
    np.savez(a.save, **{str(k): v for k, v in sig.items()})
    print(f"GOLDEN steps={a.steps} batch={a.batch} peak={peak:.2f}GiB per_step={dt:.2f}s nlayers={len(sig)}", flush=True)
