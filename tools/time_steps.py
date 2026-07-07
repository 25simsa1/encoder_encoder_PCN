"""Time the graph-compiled PC sweep, isolating the one-time tf.function trace
(step 0) from the warm per-step cost (steps 1+). The gate's --steps average
conflates the ~50s first trace with the warm steps, so this reports each step's
wall-clock separately to make the speedup claim reproducible. Runs the real
bidirectional EncoderEncoderPCN class at batch 1, both inputs clamped."""
import time
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN

m = EncoderEncoderPCN(1e-4)
img = tf.random.normal((1, 572, 572, 3), seed=0)
txt = tf.random.normal((1, 192, 512), seed=0)
mask = tf.zeros((1, 192), tf.float32)
m.img_input.is_clamped = True
m.txt_input.is_clamped = True
m.pass_through(img, txt, mask)
for i in range(5):
    t = time.time()
    m.update_states_wts_b(1)
    dt = time.time() - t
    print(f"[step {i}] {dt:.3f}s {'(build+trace+exec)' if i == 0 else '(warm)'}", flush=True)
print("TIME_STEPS_DONE", flush=True)
