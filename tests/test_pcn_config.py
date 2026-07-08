# tests/test_pcn_config.py
from pcn_config import PCNConfig, NATIVE_7B, COCO64_156M

def test_native_reproduces_current_literals():
    c = NATIVE_7B
    assert c.img_resolution == 572
    assert c.conv_channels == (64, 64, 128, 128, 256, 256, 512, 512, 1024)
    assert c.inter_dim == 100
    # tap order conv9, conv8, conv6, conv4, conv2
    assert c.img_dense_relu_widths == (307200, 582542, 1279723, 2654815, 5433667)
    assert c.shared_latent_dims == (102400, 161817, 345871, 702332, 1429912)
    assert c.txt_seq_len == 192 and c.txt_embed_dim == 512
    assert c.txt_group_widths == (512, 1024, 2048, 4096)
    assert c.txt_group_blocks == (3, 3, 3, 8)
    assert c.txt_heads == 8 and c.txt_sublayers == 3
    assert c.txt_bridge_seq_lens == (48, 12, 3)
    assert c.txt_dense_relu_widths == (36864, 44237, 90931, 185795, 373555)
    assert c.txt_tap_indices == (-1, 12, 8, 5, 2)
    assert c.conv_padding == 'VALID'

def test_tap_indices():
    assert NATIVE_7B.txt_tap_indices == (-1, 12, 8, 5, 2)
    assert COCO64_156M.txt_tap_indices == (-1, 4, 2, 1, 0)
    # every COCO64 tap index must be valid for its trunk
    n = sum(COCO64_156M.txt_group_blocks)          # 6
    assert all(-n <= i < n for i in COCO64_156M.txt_tap_indices)

def test_shared_dims_are_the_shared_contract():
    # the five shared-latent dims are what image dense2/6/10/14/18 and text
    # dense4/8/12/16/20 both use; a single tuple guarantees they match.
    assert len(NATIVE_7B.shared_latent_dims) == 5
    assert len(COCO64_156M.shared_latent_dims) == 5

def test_coco64_is_64px_and_smaller():
    c = COCO64_156M
    assert c.img_resolution == 64
    assert c.conv_channels == NATIVE_7B.conv_channels  # depth/channels unchanged
    assert c.inter_dim == 100
    # initial estimate (Task 4 tunes these); must be far smaller than native
    assert max(c.shared_latent_dims) <= max(NATIVE_7B.shared_latent_dims) // 10
    assert c.txt_seq_len == 64
    assert c.txt_embed_dim == 50
    assert c.conv_padding == 'SAME'


def test_conv_padding_is_config_driven():
    # native trunk keeps VALID (byte-identical); 64px path needs SAME so the 9
    # convs preserve spatial size and only the 4 maxpools reduce 64->4.
    assert NATIVE_7B.conv_padding == 'VALID'
    assert COCO64_156M.conv_padding == 'SAME'


def test_config_post_init_lengths():
    from pcn_config import PCNConfig, NATIVE_7B, COCO64_156M
    for c in (NATIVE_7B, COCO64_156M):
        assert len(c.conv_channels) == 9
        assert len(c.img_dense_relu_widths) == 5 == len(c.shared_latent_dims) == len(c.txt_dense_relu_widths) == len(c.txt_tap_indices)
        assert len(c.txt_group_widths) == len(c.txt_group_blocks)
        assert len(c.txt_bridge_seq_lens) == len(c.txt_group_widths) - 1


def test_downsample_field_and_coco64_gen():
    from pcn_config import PCNConfig, NATIVE_7B, COCO64_156M, COCO64_GEN
    assert NATIVE_7B.downsample == 'maxpool'
    assert COCO64_156M.downsample == 'maxpool'
    assert COCO64_GEN.downsample == 'strided_conv'
    # COCO64_GEN matches COCO64_156M in every other field
    for f in vars(COCO64_156M):
        if f != 'downsample':
            assert getattr(COCO64_GEN, f) == getattr(COCO64_156M, f)

def test_downsample_validation():
    import pytest
    from pcn_config import PCNConfig, COCO64_156M
    import dataclasses
    with pytest.raises(Exception):
        dataclasses.replace(COCO64_156M, downsample='bogus')
