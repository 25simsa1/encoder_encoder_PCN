import numpy as np
import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer
from dense_pcn_layer import DensePCNLayer


def _dense_pair(iso):
    prev = DensePCNLayer(16, 1e-3, "linear")
    L = DensePCNLayer(4, 1e-3, "linear", prev_layer=prev)   # (16,4), in > out
    x = tf.random.normal((8, 16))
    prev(x, set_state=True)
    L(prev.predict_next(), set_state=True)
    L.iso_eta = iso
    return L


def _ortho_err(w):
    m = tf.reshape(w, (-1, int(w.shape[-1])))
    return float(tf.norm(tf.linalg.matrix_transpose(m) @ m - tf.eye(int(w.shape[-1]))))


def test_iso_default_off_is_noop():
    tf.random.set_seed(0); a = _dense_pair(0.0); a.update_wts(); wa = a.wts.numpy()
    tf.random.set_seed(0); b = _dense_pair(0.0); b.update_wts(); wb = b.wts.numpy()
    np.testing.assert_array_equal(wa, wb)    # deterministic and unchanged by the (absent) iso branch
    assert a.iso_eta == 0.0


def test_iso_flow_orthogonalizes_dense():
    L = _dense_pair(0.05)
    e0 = _ortho_err(L.wts)
    for _ in range(200):
        L.update_wts()
    e1 = _ortho_err(L.wts)
    assert e1 < 0.3 * e0                      # ||WtW - I|| shrinks decisively
    cols = tf.norm(L.wts, axis=0).numpy()
    np.testing.assert_allclose(cols, np.ones_like(cols), atol=0.15)   # column norms -> 1


def test_iso_flow_orthogonalizes_conv_kernel():
    prev = Conv2DPCNLayer(3, (3, 3), 1e-3, "linear", padding="SAME")
    L = Conv2DPCNLayer(5, (3, 3), 1e-3, "linear", padding="SAME", prev_layer=prev)   # K (27,5)
    x = tf.random.normal((2, 8, 8, 3))
    prev(x, set_state=True)
    L(prev.predict_next(), set_state=True)
    L.iso_eta = 0.05
    e0 = _ortho_err(L.wts)
    for _ in range(200):
        L.update_wts()
    e1 = _ortho_err(L.wts)
    assert e1 < 0.3 * e0
