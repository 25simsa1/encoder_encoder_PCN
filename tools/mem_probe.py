"""Task 3 diagnostic: is the 34->68 GiB memory behavior a per-step RETENTION leak
(resident 'current' rises step over step) or just large TRANSIENT gradient tensors
(peak high each step, current flat)? Builds the bidirectional class, then runs 10
single relax+weight steps, resetting the peak each step so 'peak_this_step' is that
step's transient high-water mark, and printing a live-EagerTensor count. Read-only."""
import gc as _gc
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN

def mem():
    i = tf.config.experimental.get_memory_info("GPU:0")
    return i["current"] / 2**30, i["peak"] / 2**30

def live_tensors():
    n = 0
    for o in _gc.get_objects():
        try:
            if isinstance(o, tf.Tensor) or "EagerTensor" in type(o).__name__:
                n += 1
        except Exception:
            pass
    return n

m = EncoderEncoderPCN(1e-4)
img = tf.zeros((1, 572, 572, 3), tf.float32)
txt = tf.zeros((1, 192, 512), tf.float32)
mask = tf.zeros((1, 192), tf.float32)
m.img_input.is_clamped = True
m.txt_input.is_clamped = True
m.pass_through(img, txt, mask)
c, p = mem()
print(f"[after pass_through] cur={c:.2f}G peak={p:.2f}G tensors={live_tensors()}", flush=True)
for step in range(10):
    tf.config.experimental.reset_memory_stats("GPU:0")
    m.update_states_wts_b(1)
    c, p = mem()
    print(f"[step {step}] cur={c:.2f}G peak_this_step={p:.2f}G tensors={live_tensors()}", flush=True)
print("MEMPROBE_DONE", flush=True)
