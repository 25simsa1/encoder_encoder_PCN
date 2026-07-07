import numpy as np, tensorflow as tf
from infonce import infonce_grads

def test_perfect_alignment_high_acc_low_loss():
    u = tf.constant(np.eye(4, 8), dtype=tf.float32)   # 4 distinct codes
    v = tf.constant(np.eye(4, 8), dtype=tf.float32)   # identical -> matched pairs align
    du, dv, acc, loss = infonce_grads(u, v, tau=0.07)
    assert float(acc) == 1.0
    assert float(loss) < 0.1
    assert du.shape == (4, 8) and dv.shape == (4, 8)

def test_misaligned_lower_acc_grad_pulls_together():
    rng = np.random.default_rng(0)
    u = tf.constant(rng.normal(size=(6, 8)), dtype=tf.float32)
    v = tf.constant(rng.normal(size=(6, 8)), dtype=tf.float32)   # random -> unaligned
    du, dv, acc, loss = infonce_grads(u, v, tau=0.07)
    assert float(loss) > 0.5                       # random pairs are hard
    assert bool(tf.reduce_all(tf.math.is_finite(du))) and bool(tf.reduce_all(tf.math.is_finite(dv)))
    # a gradient DESCENT step on u should reduce the loss
    u2 = u - 0.5 * du
    _, _, _, loss2 = infonce_grads(u2, v, tau=0.07)
    assert float(loss2) < float(loss)
