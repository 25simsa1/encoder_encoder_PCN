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

def golden_state_signature_ex0(model):
    sig = {}
    for i, L in enumerate(model.trainable_layers):
        s = getattr(L, "state", None)
        if s is not None:
            sig[i] = float(tf.norm(tf.cast(s[0:1], tf.float32)))  # example 0 only
    return sig

def max_cross_example_dev(model):
    worst = 0.0
    for L in model.trainable_layers:
        s = getattr(L, "state", None)
        if s is None or int(s.shape[0]) < 2:
            continue
        s0 = tf.cast(s[0:1], tf.float32)
        n0 = float(tf.norm(s0)) + 1e-9
        for i in range(1, int(s.shape[0])):
            d = float(tf.norm(tf.cast(s[i:i+1], tf.float32) - s0)) / n0
            worst = max(worst, d)
    return worst

def run_reference(steps=2, batch=1, seed=0, relaxed=False, relax_steps=5, weight_steps=2, dup_batch=0):
    tf.random.set_seed(seed)
    m = EncoderEncoderPCN(1e-4)
    if dup_batch > 0:
        img1 = tf.random.normal((1, 572, 572, 3), seed=seed)
        txt1 = tf.random.normal((1, 192, 512), seed=seed)
        mask1 = tf.zeros((1, 192), tf.float32)
        img = tf.tile(img1, [dup_batch, 1, 1, 1])
        txt = tf.tile(txt1, [dup_batch, 1, 1])
        mask = tf.tile(mask1, [dup_batch, 1])
        batch = dup_batch
    else:
        img = tf.random.normal((batch, 572, 572, 3), seed=seed)
        txt = tf.random.normal((batch, 192, 512), seed=seed)
        mask = tf.zeros((batch, 192), tf.float32)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    try: tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception: pass
    t = time.time()
    if relaxed:
        m.update_states_wts_b_relaxed(weight_steps, relax_steps)
    else:
        m.update_states_wts_b(steps)
    dt = (time.time() - t) / max(1, steps)
    peak = 0.0
    try: peak = tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**30
    except Exception: pass
    if dup_batch > 0:
        maxdev = max_cross_example_dev(m)
        ok = maxdev <= 1e-5
        verdict = "DUPBATCH_CONSISTENT" if ok else "DUPBATCH_INCONSISTENT"
        print(f"{verdict} maxdev={maxdev:.3e}", flush=True)
        return golden_state_signature_ex0(m), peak, dt, ok
    return golden_state_signature(m), peak, dt, True

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
    ap.add_argument("--relaxed", action="store_true")
    ap.add_argument("--relax-steps", type=int, default=5)
    ap.add_argument("--weight-steps", type=int, default=2)
    ap.add_argument("--dup-batch", type=int, default=0)
    a = ap.parse_args()
    sig, peak, dt, ok = run_reference(a.steps, a.batch, relaxed=a.relaxed, relax_steps=a.relax_steps, weight_steps=a.weight_steps, dup_batch=a.dup_batch)
    np.savez(a.save, **{str(k): v for k, v in sig.items()})
    print(f"GOLDEN steps={a.steps} batch={a.batch} peak={peak:.2f}GiB per_step={dt:.2f}s nlayers={len(sig)} relaxed={a.relaxed} W={a.weight_steps} R={a.relax_steps} dup_batch={a.dup_batch}", flush=True)
    if a.dup_batch > 0 and not ok:
        raise SystemExit(1)
