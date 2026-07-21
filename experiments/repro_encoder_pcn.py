import tensorflow as tf
from encoder_encoder_pcn import EncoderEncoderPCN
import traceback

img = tf.zeros((1,572,572,3), dtype=tf.float32)
txt = tf.zeros((1,192,512), dtype=tf.float32)
mask = tf.zeros((1,192), dtype=tf.float32)
model = EncoderEncoderPCN(1e-4)
try:
    model.train_step(1, img, txt, mask)
    print('OK')
except Exception:
    traceback.print_exc()
