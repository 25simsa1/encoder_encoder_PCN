import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import COCO64_156M as C

def finite(m):
    bad = [i for i, L in enumerate(m.trainable_layers)
           if getattr(L, "state", None) is not None
           and (bool(tf.reduce_any(tf.math.is_nan(L.state))) or bool(tf.reduce_any(tf.math.is_inf(L.state))))]
    return bad

# 1) batched relaxed step, states finite
m = EncoderEncoderPCN(1e-4, config=C)
B = 4
img = tf.random.normal((B, C.img_resolution, C.img_resolution, 3), seed=0)
txt = tf.random.normal((B, C.txt_seq_len, C.txt_embed_dim), seed=0)
mask = tf.zeros((B, C.txt_seq_len))
m.img_input.is_clamped = True; m.txt_input.is_clamped = True
m.pass_through(img, txt, mask)
m.update_states_wts_b_relaxed(2, 5)
print(f"BATCHED_RELAXED_STEP nonfinite_layers={finite(m)}", flush=True)

# 2) shared-latent aliasing: image dense2/6/10/14/18 share state with text dense4/8/12/16/20.
# They are the layers whose .state IS another layer's .state; assert 5 aliased pairs exist.
states = [id(getattr(L, "state", None)) for L in m.trainable_layers if getattr(L, "state", None) is not None]
n_shared = len(states) - len(set(states))
print(f"SHARED_STATE_ALIASES={n_shared} (expect 5)", flush=True)

# 3) both generation directions on a fresh model
mg = EncoderEncoderPCN(1e-4, config=C)
oi = mg.test_step(10, img[:1], txt[:1], predict='img', mask=mask[:1])
ot = mg.test_step(10, img[:1], txt[:1], predict='txt', mask=mask[:1])
fi = bool(tf.reduce_all(tf.math.is_finite(oi))); ft = bool(tf.reduce_all(tf.math.is_finite(ot)))
print(f"GEN_IMG shape={tuple(oi.shape)} finite={fi}", flush=True)
print(f"GEN_TXT shape={tuple(ot.shape)} finite={ft}", flush=True)
print("VALIDATE_COCO64_DONE", flush=True)
