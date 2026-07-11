import numpy as np
import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer
from dense_pcn_layer import DensePCNLayer


def _realize_conv():
    L = Conv2DPCNLayer(5, (3, 3), 1e-2, "linear", padding="SAME")
    L(tf.random.normal((2, 8, 8, 3)), set_state=True)   # realizes wts
    return L


def test_conv_weight_off_is_wts_identity():
    L = _realize_conv()
    assert L.weight_norm is False
    assert L.weight() is L.wts            # off = passthrough, byte-identical


def test_conv_enable_is_seamless():
    L = _realize_conv()
    w_before = L.weight().numpy().copy()  # == wts (off)
    L.enable_weight_norm()
    assert L.weight_norm is True
    np.testing.assert_allclose(L.weight().numpy(), w_before, atol=1e-5)   # w == wts at enable
    per_unit = tf.sqrt(tf.reduce_sum(tf.square(L.weight()), axis=[0, 1, 2])).numpy()
    np.testing.assert_allclose(per_unit, L.g_mag.numpy(), atol=1e-4)      # ||w|| == g_mag per filter


def test_conv_enable_requires_realized_wts():
    L = Conv2DPCNLayer(5, (3, 3), 1e-2, "linear", padding="SAME")   # wts not realized
    try:
        L.enable_weight_norm()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_conv_update_preserves_wts_norm():
    prev = Conv2DPCNLayer(3, (3, 3), 1e-3, "linear", padding="SAME")
    L = Conv2DPCNLayer(5, (3, 3), 1e-3, "linear", padding="SAME", prev_layer=prev)
    x = tf.random.normal((2, 8, 8, 3))
    prev(x, set_state=True)                    # prev.state (2,8,8,3)
    L(prev.predict_next(), set_state=True)     # L.wts (3,3,3,5), L.state (2,8,8,5)
    L.enable_weight_norm()
    n0 = tf.sqrt(tf.reduce_sum(tf.square(L.wts), axis=[0, 1, 2])).numpy()
    g0 = L.g_mag.numpy().copy()
    for _ in range(5):
        L.state.assign(L.state + tf.random.normal(L.state.shape) * 0.1)   # keep the gradient nonzero
        L.update_wts()
    n1 = tf.sqrt(tf.reduce_sum(tf.square(L.wts), axis=[0, 1, 2])).numpy()
    np.testing.assert_allclose(n1, n0, rtol=0.1)          # tangential step keeps ||wts|| put
    assert np.abs(L.g_mag.numpy() - g0).max() > 1e-6      # magnitude actually moved


def _realize_dense():
    L = DensePCNLayer(3, 1e-2, "linear")
    L(tf.random.normal((8, 4)), set_state=True)   # realizes wts (4,3) and b (3,)
    return L


def test_dense_weight_off_is_wts_identity():
    L = _realize_dense()
    assert L.weight_norm is False
    assert L.weight() is L.wts


def test_dense_enable_is_seamless():
    L = _realize_dense()
    w_before = L.weight().numpy().copy()
    L.enable_weight_norm()
    assert L.weight_norm is True
    np.testing.assert_allclose(L.weight().numpy(), w_before, atol=1e-5)
    per_unit = tf.norm(L.weight(), axis=0).numpy()            # per output column
    np.testing.assert_allclose(per_unit, L.g_mag.numpy(), atol=1e-4)


def test_dense_update_preserves_wts_norm():
    # Runs the real update_wts weight-norm branch: ||wts|| per column stays put
    # (tangential direction step) while g_mag moves (radial magnitude step).
    prev = DensePCNLayer(4, 1e-3, "linear")
    L = DensePCNLayer(3, 1e-3, "linear", prev_layer=prev)
    x = tf.random.normal((8, 4))
    prev(x, set_state=True)                  # prev.state (8,4)
    L(prev.predict_next(), set_state=True)   # L.wts (4,3), L.b (3,), L.state (8,3)
    L.enable_weight_norm()
    n0 = tf.norm(L.wts, axis=0).numpy()
    g0 = L.g_mag.numpy().copy()
    for _ in range(5):
        L.state.assign(L.state + tf.random.normal(L.state.shape) * 0.1)
        L.update_wts()
    n1 = tf.norm(L.wts, axis=0).numpy()
    np.testing.assert_allclose(n1, n0, rtol=0.1)
    assert np.abs(L.g_mag.numpy() - g0).max() > 1e-6
