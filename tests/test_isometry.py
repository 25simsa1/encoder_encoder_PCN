import numpy as np
import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer
from dense_pcn_layer import DensePCNLayer


def _aniso(w):
    # relative anisotropy of the small-side Gram: ||gram/c - I|| where c = trace/n
    m = tf.reshape(w, (-1, int(w.shape[-1])))
    r, c_ = int(m.shape[0]), int(m.shape[-1])
    n = min(r, c_)
    gram = (tf.linalg.matrix_transpose(m) @ m) if c_ <= r else (m @ tf.linalg.matrix_transpose(m))
    c = tf.linalg.trace(gram) / float(n)
    return float(tf.norm(gram / (c + 1e-8) - tf.eye(n)))


def _dense_pair(din, dout, iso):
    prev = DensePCNLayer(din, 1e-3, "linear")
    L = DensePCNLayer(dout, 1e-3, "linear", prev_layer=prev)
    x = tf.random.normal((8, din))
    prev(x, set_state=True)
    L(prev.predict_next(), set_state=True)
    L.iso_eta = iso
    return L


def test_iso_default_off_is_noop():
    tf.random.set_seed(0); a = _dense_pair(16, 4, 0.0); a.update_wts(); wa = a.wts.numpy()
    tf.random.set_seed(0); b = _dense_pair(16, 4, 0.0); b.update_wts(); wb = b.wts.numpy()
    np.testing.assert_array_equal(wa, wb)
    assert a.iso_eta == 0.0


def test_iso_flow_isotropizes_contracting_dense_and_keeps_scale():
    L = _dense_pair(16, 4, 0.05)
    e0 = _aniso(L.wts); s0 = float(tf.norm(L.wts))
    for _ in range(200):
        L.update_wts()
    assert _aniso(L.wts) < 0.3 * e0                       # anisotropy shrinks decisively
    s1 = float(tf.norm(L.wts))
    assert 0.5 * s0 < s1 < 2.0 * s0                       # the overall scale is NOT collapsed


def test_iso_flow_isotropizes_expanding_dense():
    # expanding edge (out > in): Gram over the SMALL side (rows); also guards the eye(out)
    # blowup crash and the unit-target scale collapse
    L = _dense_pair(4, 16, 0.05)
    e0 = _aniso(L.wts); s0 = float(tf.norm(L.wts))
    for _ in range(200):
        L.update_wts()
    assert _aniso(L.wts) < 0.3 * e0
    s1 = float(tf.norm(L.wts))
    assert 0.5 * s0 < s1 < 2.0 * s0


def test_iso_flow_isotropizes_conv_kernel():
    prev = Conv2DPCNLayer(3, (3, 3), 1e-3, "linear", padding="SAME")
    L = Conv2DPCNLayer(5, (3, 3), 1e-3, "linear", padding="SAME", prev_layer=prev)   # K (27,5)
    x = tf.random.normal((2, 8, 8, 3))
    prev(x, set_state=True)
    L(prev.predict_next(), set_state=True)
    L.iso_eta = 0.05
    e0 = _aniso(L.wts)
    for _ in range(200):
        L.update_wts()
    assert _aniso(L.wts) < 0.3 * e0
