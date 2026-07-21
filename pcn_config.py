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
    downsample: str = 'maxpool'   # image downsampling: 'maxpool' (native, one-way) or 'strided_conv' (invertible)

    def __post_init__(self):
        assert len(self.conv_channels) == 9, "expected 9 conv channels"
        assert len(self.img_dense_relu_widths) == len(self.shared_latent_dims) == len(self.txt_dense_relu_widths) == len(self.txt_tap_indices) == 5, "expected 5 per-scale values"
        assert len(self.txt_group_widths) == len(self.txt_group_blocks), "group widths/blocks length mismatch"
        assert len(self.txt_bridge_seq_lens) == len(self.txt_group_widths) - 1, "expected one bridge per group gap"
        assert self.downsample in ('maxpool', 'strided_conv'), f"downsample must be maxpool|strided_conv, got {self.downsample}"

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

# Task 4: measured 107.3M at the initial widths below (3x too small), so
# img_dense_relu_widths and shared_latent_dims are scaled by 3x from the
# original doubling series (2048,4096,8192,16384,32768) to land near 156M.
# Shared dims stay an ~8x compression of each 64px tap feature map, keeping
# the native ordering (bigger latent for the shallow high-res taps).
COCO64_156M = PCNConfig(
    name="coco64",
    img_resolution=64,
    conv_channels=(64, 64, 128, 128, 256, 256, 512, 512, 1024),
    inter_dim=100,
    img_dense_relu_widths=(6144, 12288, 24576, 49152, 98304),
    shared_latent_dims=(6144, 12288, 24576, 49152, 98304),
    txt_seq_len=64, txt_embed_dim=50,   # 50 = coco64_data.V (one-hot char), 64 = seq
    txt_group_widths=(512, 512, 512, 512),
    txt_group_blocks=(1, 1, 1, 3),
    txt_heads=8, txt_sublayers=3,
    txt_bridge_seq_lens=(16, 8, 4),
    txt_dense_relu_widths=(2048, 4096, 8192, 8192, 8192),
    txt_tap_indices=(-1, 4, 2, 1, 0),    # final, mid-group3, group2/1/0 ends for the 6-block 1+1+1+3 trunk
    conv_padding='SAME',                 # 64px input: convs preserve spatial size, only the 4 maxpools reduce 64->32->16->8->4
)

# Task 2: opt-in invertible downsampling for the top-down generative path.
# Same as COCO64_156M but swaps the one-way maxpools for stride-2 shared-weight
# convs (conv2d down / transpose-conv up), letting the generative drive flow
# top-down through the downsampling stages instead of being blocked by it.
import dataclasses as _dc
COCO64_GEN = _dc.replace(COCO64_156M, downsample='strided_conv')

# Wide-inter escalation: the ridge experiment (LOG 2026-07-16) proved the
# per-edge-linear ceiling at inter_dim=100 — exact closed-form optimal
# top-down inverses still decode to a shared attractor because the five
# downward inter pipes carry only 57-72% of the below-layer variance each.
# 512 per tap (2560 total downward rank) clears the PCA-500
# recognizable-blurry ceiling with margin. New name = new checkpoints;
# NATIVE and COCO64_GEN are untouched.
COCO64_WIDE = _dc.replace(COCO64_GEN, inter_dim=512, name="coco64w")
