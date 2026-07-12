import numpy as np
import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer


def _relaxing_layer(noise):
    prev = Conv2DPCNLayer(3, (3, 3), 1e-1, "linear", padding="SAME")
    L = Conv2DPCNLayer(5, (3, 3), 1e-1, "linear", padding="SAME", prev_layer=prev)
    x = tf.random.normal((2, 8, 8, 3))
    prev(x, set_state=True)
    L(prev.predict_next(), set_state=True)
    prev.is_clamped = True          # clamp prev so L actually relaxes toward it
    L.noise_temp = noise
    return L


def test_noise_default_is_zero():
    L = Conv2DPCNLayer(5, (3, 3), 1e-3, "linear", padding="SAME")
    assert L.noise_temp == 0.0


def test_noise_off_is_deterministic():
    # noise_temp=0 => update_state takes no random draw, so it is byte-reproducible
    tf.random.set_seed(0); a = _relaxing_layer(0.0); s0 = a.state.numpy().copy(); a.update_state(); da = a.state.numpy() - s0
    tf.random.set_seed(0); b = _relaxing_layer(0.0); s0b = b.state.numpy().copy(); b.update_state(); db = b.state.numpy() - s0b
    np.testing.assert_array_equal(da, db)


def test_noise_on_perturbs_the_state():
    # same build + same deterministic update, but noise_temp>0 adds a Langevin draw
    tf.random.set_seed(0); off = _relaxing_layer(0.0); off.update_state(); s_off = off.state.numpy()
    tf.random.set_seed(0); on = _relaxing_layer(1.0);  on.update_state();  s_on = on.state.numpy()
    assert np.abs(s_on - s_off).max() > 1e-3     # the noise moved the state off the deterministic point
