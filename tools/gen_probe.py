"""M4 diagnosis: is the generation path's "double update" of the unclamped
input intentional or a bug?

`update_states_img`/`update_states_txt` (used by `test_step`, the only
generation entry point) do, per step:
    for layer in self.trainable_layers:   # img_input IS in this list -> updated once here
        layer.update_state()
    self.img_input.update_state()          # img_input updated AGAIN

The suspected reason: img_input/txt_input sit at INDEX 0 of `trainable_layers`
(each is appended before its own downstream chain is built), so the per-layer
sweep updates them FIRST, using last step's stale downstream state, then only
updates the downstream chain afterward using the freshly-updated input. The
explicit second call re-syncs the input to the latents just refreshed THIS
step, avoiding a one-step lag. This script confirms or refutes that with a
direct ablation, on a FRESH untrained model (random weights - this is a
mechanism test, not an image-quality test), plus checks finiteness/
convergence of both generation directions and roughly times the uncompiled
eager path.

Read-only: does not modify encoder_encoder_pcn.py or any *_pcn_layer.py. Uses
only the class's public methods (test_step, pass_through, predict_next) plus
calling layer.update_state() directly (a public per-layer method) to drive
the manual/ablation loops.

Each full model instance is ~30-70GiB (see docs/superpowers/plans/
2026-07-06-bidirectional-execution-rewrite.md Task 3 findings), and building
several in the SAME process (even freed with del+gc.collect() between them)
fragmented the GPU BFC allocator enough to OOM the 4th/5th build in one run.
So --part lets each part run as its own fresh process (its own clusterrun.sh
invocation) with a clean allocator; --part all runs everything in one
process for convenience when memory allows.
"""
import argparse
import gc
import time
import tensorflow as tf
import numpy as np

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN

NUM_STEPS = 15
IMG_SHAPE = (1, 572, 572, 3)
TXT_SHAPE = (1, 192, 512)
MASK_SHAPE = (1, 192)


def fresh_model(seed=0):
    # Reset the global RNG right before construction so every model built in
    # this script (weights are drawn via bare tf.random.normal in the layer
    # ctors, no per-call seed) gets IDENTICAL initial weights. That isolates
    # the ablation to the one line under test, not to different random inits.
    tf.random.set_seed(seed)
    return EncoderEncoderPCN(1e-4)


def all_states_finite(model):
    bad = []
    for i, L in enumerate(model.trainable_layers):
        s = getattr(L, "state", None)
        if s is None:
            continue
        if not bool(tf.reduce_all(tf.math.is_finite(tf.cast(s, tf.float32)))):
            bad.append(i)
    return bad


def stats(t):
    t = tf.cast(t, tf.float32)
    return float(tf.reduce_min(t)), float(tf.reduce_max(t)), float(tf.reduce_mean(t))


def free(m):
    del m
    gc.collect()


def get_inputs():
    img = tf.random.normal(IMG_SHAPE, seed=0)
    txt = tf.random.normal(TXT_SHAPE, seed=0)
    mask = tf.zeros(MASK_SHAPE, tf.float32)
    return img, txt, mask


def part1(img, txt, mask):
    print("=== Part 1: both directions run + finiteness (via the real test_step) ===", flush=True)

    m1 = fresh_model(0)
    t0 = time.time()
    out_img = m1.test_step(NUM_STEPS, img, txt, predict='img', mask=mask)
    dt_img = time.time() - t0
    finite_out_img = bool(tf.reduce_all(tf.math.is_finite(tf.cast(out_img, tf.float32))))
    bad_img = all_states_finite(m1)
    mn, mx, me = stats(out_img)
    print(f"[predict=img] wall={dt_img:.2f}s shape={tuple(out_img.shape)} "
          f"ASSERT_FINITE_OUTPUT={'PASS' if finite_out_img else 'FAIL'} "
          f"ASSERT_FINITE_ALL_STATES={'PASS' if not bad_img else 'FAIL(bad_layers=' + str(bad_img) + ')'} "
          f"min={mn:.6f} max={mx:.6f} mean={me:.6f}", flush=True)
    free(m1)

    m2 = fresh_model(0)
    t0 = time.time()
    out_txt = m2.test_step(NUM_STEPS, img, txt, predict='txt', mask=mask)
    dt_txt = time.time() - t0
    finite_out_txt = bool(tf.reduce_all(tf.math.is_finite(tf.cast(out_txt, tf.float32))))
    bad_txt = all_states_finite(m2)
    mn2, mx2, me2 = stats(out_txt)
    print(f"[predict=txt] wall={dt_txt:.2f}s shape={tuple(out_txt.shape)} "
          f"ASSERT_FINITE_OUTPUT={'PASS' if finite_out_txt else 'FAIL'} "
          f"ASSERT_FINITE_ALL_STATES={'PASS' if not bad_txt else 'FAIL(bad_layers=' + str(bad_txt) + ')'} "
          f"min={mn2:.6f} max={mx2:.6f} mean={me2:.6f}", flush=True)
    free(m2)


def part2(img, txt, mask):
    print("\n=== Part 2: convergence of the unclamped img_input over 15 steps ===", flush=True)
    print("(manual loop replicating the REAL generation path exactly: per-layer "
          "sweep + the explicit second img_input.update_state(), instrumented "
          "between steps)", flush=True)

    m3 = fresh_model(0)
    m3.pass_through(img, txt, mask)
    m3.img_input.is_clamped = False
    m3.txt_input.is_clamped = True
    conv_norms = []
    step_times = []
    prev = None
    for step in range(NUM_STEPS):
        ts = time.time()
        for L in m3.trainable_layers:
            L.update_state()
        m3.img_input.update_state()
        step_times.append(time.time() - ts)
        cur = tf.identity(m3.img_input.state)
        norm = float(tf.norm(tf.cast(cur, tf.float32)))
        conv_norms.append(norm)
        if prev is None:
            rel = float('nan')
        else:
            rel = float(tf.norm(tf.cast(cur - prev, tf.float32))) / (float(tf.norm(tf.cast(prev, tf.float32))) + 1e-9)
        print(f"[conv step {step:2d}] norm={norm:.6f} rel_change={rel:.6f} step_time={step_times[-1]:.3f}s", flush=True)
        prev = cur
    mean_step = float(np.mean(step_times))
    mean_step_warm = float(np.mean(step_times[1:])) if NUM_STEPS > 1 else float('nan')
    print(f"[conv] mean_step_time={mean_step:.3f}s mean_step_time_excl_first={mean_step_warm:.3f}s "
          f"(uncompiled/eager path, no tf.function)", flush=True)
    free(m3)
    return conv_norms


def run_ablation(img, txt, mask, with_second_update, seed=0):
    m = fresh_model(seed)
    m.pass_through(img, txt, mask)
    m.img_input.is_clamped = False
    m.txt_input.is_clamped = True
    norms = []
    for step in range(NUM_STEPS):
        for L in m.trainable_layers:
            L.update_state()
        if with_second_update:
            m.img_input.update_state()
        norms.append(float(tf.norm(tf.cast(m.img_input.state, tf.float32))))
    final_out = m.img_input.predict_next().numpy().copy()
    free(m)
    return norms, final_out


def part3(img, txt, mask, conv_norms=None):
    print("\n=== Part 3: ablation of the second (explicit) img_input.update_state() call ===", flush=True)

    norms_with, out_with = run_ablation(img, txt, mask, True, seed=0)
    norms_without, out_without = run_ablation(img, txt, mask, False, seed=0)

    if conv_norms is not None:
        # Sanity check: Part 2's manual loop is procedurally identical to the
        # WITH ablation arm (same seed, same inputs, same update order) ->
        # curves should match to floating-point precision. Cross-checks both
        # loops are wired the same way as the real generation path.
        sanity_max_dev = max(abs(a - b) for a, b in zip(conv_norms, norms_with))
        print(f"[sanity] Part2 vs ablation-WITH norm curves max abs dev = {sanity_max_dev:.3e} "
              f"(expect ~0, same procedure/seed)", flush=True)

    rel_diff = float(np.linalg.norm(out_with - out_without) / (np.linalg.norm(out_with) + 1e-9))
    finite_with = bool(np.all(np.isfinite(out_with)))
    finite_without = bool(np.all(np.isfinite(out_without)))
    print(f"[ablation] final predict_next() relative L2 diff (WITH vs WITHOUT) = {rel_diff:.6f}", flush=True)
    print(f"[ablation] finite WITH={finite_with} finite WITHOUT={finite_without}", flush=True)

    print("[ablation] convergence curves (img_input state L2 norm per step):", flush=True)
    print(" step |    WITH_norm  WITH_relchg |  WITHOUT_norm  WITHOUT_relchg", flush=True)
    prev_w = prev_wo = None
    for i in range(NUM_STEPS):
        w, wo = norms_with[i], norms_without[i]
        rw = float('nan') if prev_w is None else abs(w - prev_w) / (prev_w + 1e-9)
        rwo = float('nan') if prev_wo is None else abs(wo - prev_wo) / (prev_wo + 1e-9)
        print(f" {i:3d}  |  {w:11.6f}  {rw:10.6f} |  {wo:11.6f}  {rwo:12.6f}", flush=True)
        prev_w, prev_wo = w, wo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["1", "2", "3", "all"],
                     help="Run only this part as an isolated process (avoids GPU "
                          "allocator fragmentation from building several full "
                          "model instances back-to-back in one process). "
                          "'all' runs everything in one process.")
    args = ap.parse_args()

    img, txt, mask = get_inputs()
    conv_norms = None
    if args.part in ("1", "all"):
        part1(img, txt, mask)
    if args.part in ("2", "all"):
        conv_norms = part2(img, txt, mask)
    if args.part in ("3", "all"):
        part3(img, txt, mask, conv_norms)

    print("\nGEN_PROBE_DONE", flush=True)
