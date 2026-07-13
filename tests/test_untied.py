import numpy as np
import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer
from dense_pcn_layer import DensePCNLayer


def _dense_chain():
    prev = DensePCNLayer(6, 1e-2, "linear")
    L = DensePCNLayer(4, 1e-2, "linear", prev_layer=prev)
    x = tf.random.normal((8, 6))
    prev(x, set_state=True)
    L(prev.predict_next(), set_state=True)
    return prev, L


def test_untied_default_off_is_tied():
    _, L = _dense_chain()
    assert L.untied is False and L.wts_td is None
    assert L.weight_td() is L.weight()          # tied fallback, byte-identical


def test_enable_untied_is_seamless():
    _, L = _dense_chain()
    pp_before = L.predict_prev().numpy()
    L.enable_untied()
    assert L.untied is True
    np.testing.assert_array_equal(L.predict_prev().numpy(), pp_before)   # copy => unchanged at enable


def test_untied_duties_diverge_and_bottom_up_is_untouched_by_dpred():
    # with the layer's own state perturbed (a top-down error) and prev CLAMPED (no d_state),
    # only wts_td moves; wts stays put. The tug-of-war is gone by construction.
    prev, L = _dense_chain()
    L.enable_untied()
    prev.is_clamped = True
    L.state.assign(L.state + tf.random.normal(L.state.shape))
    w0 = L.wts.numpy().copy(); t0 = L.wts_td.numpy().copy()
    for _ in range(3):
        L.update_wts()
    np.testing.assert_array_equal(L.wts.numpy(), w0)          # bottom-up weight untouched
    assert np.abs(L.wts_td.numpy() - t0).max() > 1e-7         # top-down weight learned


def test_untied_conv_duties_split():
    prev = Conv2DPCNLayer(3, (3, 3), 1e-2, "linear", padding="SAME")
    L = Conv2DPCNLayer(5, (3, 3), 1e-2, "linear", padding="SAME", prev_layer=prev)
    x = tf.random.normal((2, 8, 8, 3))
    prev(x, set_state=True)
    L(prev.predict_next(), set_state=True)
    L.enable_untied()
    prev.is_clamped = True
    L.state.assign(L.state + tf.random.normal(L.state.shape))
    w0 = L.wts.numpy().copy(); t0 = L.wts_td.numpy().copy()
    for _ in range(3):
        L.update_wts()
    np.testing.assert_array_equal(L.wts.numpy(), w0)
    assert np.abs(L.wts_td.numpy() - t0).max() > 1e-7
