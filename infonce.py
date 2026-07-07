"""PC-native InfoNCE: symmetric contrastive loss over a batch of paired codes, with
gradients taken ONLY w.r.t. the codes (a GradientTape scoped to codes->loss; it never
touches network weights). Used to inject a coupling error into PC relaxation."""
import tensorflow as tf

def infonce_grads(u, v, tau=0.07):
    u = tf.convert_to_tensor(u); v = tf.convert_to_tensor(v)
    with tf.GradientTape() as t:
        t.watch([u, v])
        un = tf.math.l2_normalize(u, axis=1)
        vn = tf.math.l2_normalize(v, axis=1)
        logits = tf.matmul(un, vn, transpose_b=True) / tau      # (B,B)
        B = tf.shape(logits)[0]
        labels = tf.range(B)
        loss = 0.5 * (
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
            + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tf.transpose(logits))))
    du, dv = t.gradient(loss, [u, v])
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels), tf.float32))
    return du, dv, acc, loss
