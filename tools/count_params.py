import argparse
import tensorflow as tf, numpy as np
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN
from pcn_config import NATIVE_7B, COCO64_156M

CFG = {"native7b": NATIVE_7B, "coco64": COCO64_156M}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="coco64")
    a = ap.parse_args(); cfg = CFG[a.config]
    r = cfg.img_resolution
    m = EncoderEncoderPCN(1e-4, config=cfg)
    img = tf.zeros((1, r, r, 3)); txt = tf.zeros((1, cfg.txt_seq_len, cfg.txt_embed_dim)); mask = tf.zeros((1, cfg.txt_seq_len))
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)   # realize lazy weight Variables
    P = 0
    for L in m.trainable_layers:
        for k, v in vars(L).items():
            if isinstance(v, tf.Variable) and k != "state":
                P += int(np.prod(v.shape))
    print(f"TOTAL_PARAMS={P} ({P/1e6:.1f}M) config={cfg.name} nlayers={len(m.trainable_layers)}", flush=True)
