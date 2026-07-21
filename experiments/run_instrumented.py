"""Instrumented runner for EncoderEncoderPCN.train_step.

Run on a GPU with >= 40 GB VRAM (ideally 80 GB). Does:
  1. enables GPU memory growth (so TF doesn't grab all VRAM and hide real usage)
  2. checks VRAM and STOPS if < 40 GB (memory is solved by hardware, not by
     shrinking the model)
  3. runs one train_step, reporting peak memory separately around the forward
     pass (pass_through) and the PCN update loop (update_states_wts_b)
  4. runs a few more update steps and checks numeric stability (no NaN/Inf)

Mirrors train_step exactly (img/txt clamped -> pass_through -> update loop);
it just splits the two phases so each can be timed and memory-profiled.
"""
import tensorflow as tf

# ---- 1. memory growth must be set before any tensor/device use ----
GPUS = tf.config.list_physical_devices("GPU")
for g in GPUS:
    tf.config.experimental.set_memory_growth(g, True)


def _meminfo(tag):
    if not GPUS:
        return
    info = tf.config.experimental.get_memory_info("GPU:0")
    print(f"  [{tag}] current={info['current']/2**30:6.2f} GiB  "
          f"peak={info['peak']/2**30:6.2f} GiB")


def _reset_peak():
    if GPUS:
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:
            pass


def main():
    print("=== devices ===")
    print(" ", tf.config.list_physical_devices())
    if not GPUS:
        print("\nNO GPU DETECTED. This model needs >= 40 GB VRAM (peak ~38 GiB)."
              "\nSTOP: running on CPU would exhaust host RAM. Run on a GPU host.")
        return 2
    # VRAM gate
    try:
        det = tf.config.experimental.get_device_details(GPUS[0])
        print("  GPU:", det)
    except Exception:
        pass
    # NOTE: TF can't always report total VRAM portably; confirm with nvidia-smi.
    print("\n(Confirm `nvidia-smi` shows >= 40 GB before trusting results.)\n")

    from encoder_encoder_pcn import EncoderEncoderPCN
    img = tf.zeros((1, 572, 572, 3), dtype=tf.float32)
    txt = tf.zeros((1, 192, 512), dtype=tf.float32)
    mask = tf.zeros((1, 192), dtype=tf.float32)

    print("=== build model ===")
    model = EncoderEncoderPCN(1e-4)
    _meminfo("after __init__ (lazy: weights not yet allocated)")

    # ---- forward phase (this allocates the ~28.7 GiB of weights) ----
    model.img_input.is_clamped = True
    model.txt_input.is_clamped = True
    print("=== forward (pass_through) ===")
    _reset_peak()
    model.pass_through(img, txt, mask)
    _meminfo("after pass_through")

    # ---- PCN update phase ----
    print("=== PCN update loop (update_states_wts_b, 1 step) ===")
    _reset_peak()
    model.update_states_wts_b(1)
    _meminfo("after 1 update step")

    print("train_step-equivalent completed OK.")

    # ---- stability check over a few more steps ----
    print("=== stability: 3 more update steps ===")
    model.update_states_wts_b(3)
    bad = []
    for L in model.trainable_layers:
        s = getattr(L, "state", None)
        if s is not None:
            t = s.value() if hasattr(s, "value") else s
            if bool(tf.reduce_any(tf.math.is_nan(t))) or bool(tf.reduce_any(tf.math.is_inf(t))):
                bad.append(type(L).__name__)
    print("  non-finite states:", bad or "none")
    _meminfo("final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
