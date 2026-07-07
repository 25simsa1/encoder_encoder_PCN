import numpy as np
import coco64_data as D

def test_vocab_size_and_specials():
    assert D.V == 50 and len(D.VOCAB) == 50
    assert D.VOCAB[0] == "<pad>" and D.VOCAB[49] == "<unk>"

def test_encode_shape_and_onehot():
    oh = D.encode_caption("a cat.")
    assert oh.shape == (64, 50)
    assert np.allclose(oh.sum(axis=1)[:6], 1.0)      # first 6 positions are one-hot
    assert np.allclose(oh[6:].sum(axis=1), 1.0)      # pad positions are one-hot on <pad>
    assert oh[6:, 0].all()                           # ...specifically index 0 = <pad>

def test_roundtrip_decode():
    assert D.decode(D.encode_caption("a cat.")) == "a cat."

def test_unknown_char_maps_to_unk():
    oh = D.encode_caption("aéb")                # non-ascii -> <unk>
    assert oh[1, 49] == 1.0                          # position 1 is <unk>

def test_truncation_to_64():
    long = "x" * 100
    oh = D.encode_caption(long)
    assert oh.shape == (64, 50)
    assert D.decode(oh) == "x" * 64

def test_mask_zero_then_negative():
    m = D.caption_mask("a cat.")                     # 6 real chars
    assert (m[:6] == 0.0).all()
    assert (m[6:] < -1e8).all()
