from dataclasses import dataclass

@dataclass(frozen=True)
class PCNConfig:
    name: str
    img_resolution: int
    conv_channels: tuple      # conv1..conv9 output channels
    inter_dim: int            # bottleneck width (the inter(100) layers)
    img_dense_relu_widths: tuple   # 5, tap order conv9,conv8,conv6,conv4,conv2
    shared_latent_dims: tuple      # 5, same order; shared image<->text
    txt_seq_len: int
    txt_embed_dim: int
    txt_group_widths: tuple   # transformer width per group
    txt_group_blocks: tuple   # transformer blocks per group
    txt_heads: int
    txt_sublayers: int        # the first arg to TransformerPCNLayer
    txt_bridge_seq_lens: tuple     # linear_2/4/6 sequence reductions between groups
    txt_dense_relu_widths: tuple   # 5, text tap dense_relu (dense3/7/11/15/19)
    txt_tap_indices: tuple    # 5, txt_transformers block index per tap
                              # (dense3,dense7,dense11,dense15,dense19); -1 = final block
    conv_padding: str         # conv op padding: 'VALID' (native, shrinks) or 'SAME' (coco64, preserves spatial)

NATIVE_7B = PCNConfig(
    name="native7b",
    img_resolution=572,
    conv_channels=(64, 64, 128, 128, 256, 256, 512, 512, 1024),
    inter_dim=100,
    img_dense_relu_widths=(307200, 582542, 1279723, 2654815, 5433667),
    shared_latent_dims=(102400, 161817, 345871, 702332, 1429912),
    txt_seq_len=192, txt_embed_dim=512,
    txt_group_widths=(512, 1024, 2048, 4096),
    txt_group_blocks=(3, 3, 3, 8),
    txt_heads=8, txt_sublayers=3,
    txt_bridge_seq_lens=(48, 12, 3),
    txt_dense_relu_widths=(36864, 44237, 90931, 185795, 373555),
    txt_tap_indices=(-1, 12, 8, 5, 2),   # current attachments; -1 == block 16 for the 17-block trunk
    conv_padding='VALID',                # native trunk sized for 572px; VALID keeps byte-identical behavior
)

# Initial estimate (~160M); Task 4 tunes to ~156M. Shared dims are an ~8x
# compression of each 64px tap feature map, keeping the native ordering
# (bigger latent for the shallow high-res taps).
COCO64_156M = PCNConfig(
    name="coco64",
    img_resolution=64,
    conv_channels=(64, 64, 128, 128, 256, 256, 512, 512, 1024),
    inter_dim=100,
    img_dense_relu_widths=(2048, 4096, 8192, 16384, 32768),
    shared_latent_dims=(2048, 4096, 8192, 16384, 32768),
    txt_seq_len=32, txt_embed_dim=512,
    txt_group_widths=(512, 512, 512, 512),
    txt_group_blocks=(1, 1, 1, 3),
    txt_heads=8, txt_sublayers=3,
    txt_bridge_seq_lens=(16, 8, 4),
    txt_dense_relu_widths=(2048, 4096, 8192, 8192, 8192),
    txt_tap_indices=(-1, 4, 2, 1, 0),    # final, mid-group3, group2/1/0 ends for the 6-block 1+1+1+3 trunk
    conv_padding='SAME',                 # 64px input: convs preserve spatial size, only the 4 maxpools reduce 64->32->16->8->4
)
