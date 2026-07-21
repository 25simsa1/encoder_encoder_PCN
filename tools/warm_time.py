"""Warm-step timing for the tf.function-compiled PC sweep (Task 5).

The golden gate's per_step averages over only 2 steps, so it is dominated by the
one-time graph trace of the ~143-layer unrolled sweep. This isolates the warm
(post-trace) per-step time and confirms the sweep is traced exactly once.
Not part of the gate; timing only."""
import os, time
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN

WARM = int(os.environ.get("WARM_STEPS", "5"))
tf.random.set_seed(0)
m = EncoderEncoderPCN(1e-4)
img = tf.random.normal((1, 572, 572, 3), seed=0)
txt = tf.random.normal((1, 192, 512), seed=0)
mask = tf.zeros((1, 192), tf.float32)
m.img_input.is_clamped = True
m.txt_input.is_clamped = True
m.pass_through(img, txt, mask)

t = time.time()
m.update_states_wts_b(1)           # first step: trace + 1 execution
warmup = time.time() - t
print(f"WARMUP trace+1step={warmup:.2f}s traces={m._sweep_trace_count}", flush=True)

t = time.time()
m.update_states_wts_b(WARM)        # all warm executions, graph reused
warm_per_step = (time.time() - t) / WARM
print(f"WARM per_step={warm_per_step:.3f}s over {WARM} steps traces={m._sweep_trace_count}", flush=True)
