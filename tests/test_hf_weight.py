import numpy as np
import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer


def test_hf_off_returns_error_unchanged():
    L = Conv2DPCNLayer(3, (3, 3), 1e-3, "linear", padding="SAME")
    assert L.hf_gamma == 0.0
    e = tf.random.normal((2, 8, 8, 3))
    np.testing.assert_array_equal(L._hf(e).numpy(), e.numpy())   # off = identity, byte-identical


def test_hf_leaves_smooth_error_but_boosts_edges():
    L = Conv2DPCNLayer(3, (3, 3), 1e-3, "linear", padding="SAME")
    L.hf_gamma = 1.0
    smooth = tf.ones((1, 8, 8, 3))                                # constant -> Laplacian ~ 0
    np.testing.assert_allclose(L._hf(smooth).numpy(), smooth.numpy(), atol=1e-4)
    sharp = np.zeros((1, 8, 8, 3), np.float32); sharp[0, 4, 4, :] = 1.0   # impulse -> big Laplacian
    out = L._hf(tf.constant(sharp)).numpy()
    assert np.abs(out - sharp).max() > 0.5                        # the edge is boosted


def _pair(hf):
    prev = Conv2DPCNLayer(3, (3, 3), 1e-2, "linear", padding="SAME")
    L = Conv2DPCNLayer(5, (3, 3), 1e-2, "linear", padding="SAME", prev_layer=prev)
    x = tf.random.normal((2, 8, 8, 3), seed=0)
    prev(x, set_state=True)
    L(prev.predict_next(), set_state=True)
    L.hf_gamma = hf
    return L


def test_hf_changes_the_weight_step():
    tf.random.set_seed(0); off = _pair(0.0); w0 = off.wts.numpy().copy(); off.update_wts(); d_off = off.wts.numpy() - w0
    tf.random.set_seed(0); on = _pair(2.0);  w1 = on.wts.numpy().copy();  on.update_wts();  d_on = on.wts.numpy() - w1
    assert np.abs(d_off).max() > 0                       # a real update happened
    assert np.abs(d_on - d_off).max() > 1e-6             # the boost changed it
