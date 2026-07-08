import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer

def _forward(L, x):
    return L(tf.convert_to_tensor(x), set_state=True)

def test_stride2_halves_and_transpose_restores():
    x = tf.random.normal((2, 8, 8, 3))
    L = Conv2DPCNLayer(5, (2, 2), 1e-4, 'linear', padding='SAME', stride=2)
    out = _forward(L, x)
    assert tuple(out.shape) == (2, 4, 4, 5)          # stride-2 SAME halves H,W
    pp = L.predict_prev()
    assert tuple(pp.shape) == (2, 8, 8, 3)           # transpose restores input H,W,C

def test_stride1_default_same_padding_preserves():
    x = tf.random.normal((2, 8, 8, 3))
    L = Conv2DPCNLayer(5, (3, 3), 1e-4, 'linear', padding='SAME')   # default stride=1
    out = _forward(L, x)
    assert tuple(out.shape) == (2, 8, 8, 5)
    pp = L.predict_prev()
    assert tuple(pp.shape) == (2, 8, 8, 3)
    assert L.stride == 1
