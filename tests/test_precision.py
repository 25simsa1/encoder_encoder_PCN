import numpy as np
import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer


def _chain():
    # prev -> L -> nxt, all realized, all unclamped (so both drive blocks are active on L)
    prev = Conv2DPCNLayer(3, (3, 3), 1e-2, "linear", padding="SAME")
    L = Conv2DPCNLayer(4, (3, 3), 1e-2, "linear", padding="SAME", prev_layer=prev)
    nxt = Conv2DPCNLayer(5, (3, 3), 1e-2, "linear", padding="SAME", prev_layer=L)
    L.next_layers = [nxt]
    x = tf.random.normal((2, 8, 8, 3))
    prev(x, set_state=True)
    L(prev.predict_next(), set_state=True)
    nxt(L.predict_next(), set_state=True)
    return prev, L, nxt


def _delta(L, s0):
    L.state.assign(s0)
    L.update_state()
    return L.state.numpy() - s0.numpy()


def test_defaults_are_one_and_both_drives_matter():
    prev, L, nxt = _chain()
    assert L.pi_td == 1.0 and L.pi_bu == 1.0
    s0 = tf.constant(L.state.numpy())
    d_base = _delta(L, s0)
    prev.state.assign(prev.state + 1.0)          # change the content BELOW
    d_prev = _delta(L, s0)
    assert np.abs(d_prev - d_base).max() > 1e-7  # bottom-up drive active by default
    prev.state.assign(prev.state - 1.0)
    nxt.state.assign(nxt.state + 1.0)            # change the content ABOVE
    d_nxt = _delta(L, s0)
    assert np.abs(d_nxt - d_base).max() > 1e-7   # top-down drive active by default


def test_pi_bu_zero_is_invariant_to_below():
    prev, L, nxt = _chain()
    L.pi_bu = 0.0
    s0 = tf.constant(L.state.numpy())
    d_a = _delta(L, s0)
    prev.state.assign(prev.state + tf.random.normal(prev.state.shape))   # scramble below
    d_b = _delta(L, s0)
    np.testing.assert_array_equal(d_a, d_b)      # exactly invariant to the content below
    nxt.state.assign(nxt.state + 1.0)
    d_c = _delta(L, s0)
    assert np.abs(d_c - d_a).max() > 1e-7        # still driven from above


def test_pi_td_zero_is_invariant_to_above():
    prev, L, nxt = _chain()
    L.pi_td = 0.0
    s0 = tf.constant(L.state.numpy())
    d_a = _delta(L, s0)
    nxt.state.assign(nxt.state + tf.random.normal(nxt.state.shape))      # scramble above
    d_b = _delta(L, s0)
    np.testing.assert_array_equal(d_a, d_b)      # exactly invariant to the content above
    prev.state.assign(prev.state + 1.0)
    d_c = _delta(L, s0)
    assert np.abs(d_c - d_a).max() > 1e-7        # still driven from below
