# Design: Phase 3 retarget the bidirectional PCN from 572px to 64px (config-driven)

Date: 2026-07-06

## Goal

Make `EncoderEncoderPCN` construct from a configuration object instead of hardcoded
literals, ship a config that reproduces the current 572px/7.7B model exactly, and add
a 64px COCO config that lands near 156M parameters. This is the prerequisite to
training on COCO-64 (Phase 4). The retarget is a resize, not a redesign, keeping the
bidirectional structure and the five-scale shared-latent wiring intact.

## Hard constraints (inherited)

- The model is the bidirectional class only (`encoder_encoder_pcn.py` plus the
  `*_pcn_layer.py` layers). One conv image network used both directions via
  `predict_prev`/`predict_next` with shared weights, one transformer text network the
  same, joined by five shared latent states. Learning stays predictive coding.
- No functional-version code, no backprop, no pretrained components.
- The five shared-latent pairs must remain aliased state Variables. Image
  `dense2/6/10/14/18` share state with text `dense4/8/12/16/20`, so within a scale the
  image side and text side MUST have the same shared-latent dimension.

## Size decision (user, 2026-07-06)

Quality-driven. Start near 156M and grow widths only if generation is not
recognizable. This is why the constructor becomes config-driven, so growing is a
config edit and the paper's capacity axis on the bidirectional class becomes a set of
configs rather than a rewrite.

## Current architecture (measured, the thing being resized)

Image path is VGG-style: `conv1..conv9` with channel schedule
64,64,128,128,256,256,512,512,1024 and four 2x2 maxpools after conv2, conv4, conv6,
conv8. Five multi-scale taps at conv9, conv8, conv6, conv4, conv2. Each tap is
`flatten -> inter(100) -> dense_relu(N) -> inter(100) -> dense_shared(M)`.

At 572px the billion-parameter cost is in the `flatten -> inter(100)` layers on the
shallow taps (conv2 flattens to ~20.9M elements, times the 100 bottleneck = ~2.06B).
The 572px dense widths (hand-picked, resolution-independent):

- dense_relu widths (image): 307200, 582542, 1279723, 2654815, 5433667
- shared-latent dims: 102400, 161817, 345871, 702332, 1429912
  (image dense2/6/10/14/18 == text dense4/8/12/16/20)

Text path is a continuous `(192, 512)` input to an embedding, positional encoding, and
17 `TransformerPCNLayer` blocks whose width climbs 512, 1024, 2048, 4096 over the
stack (8 heads, 3 sublayers each), with five taps mirroring the image shared-latent
dims. This text ladder is itself multi-billion-parameter.

## Section 1: config-driven constructor

Introduce a `PCNConfig` (a small dataclass or plain dict) that parameterizes
everything that varies with size:

- `img_resolution` (default 572 for native, 64 for coco)
- conv channel schedule and which conv layers are tapped (default keeps the current
  9-conv, 4-maxpool, 5-tap structure)
- `inter_dim` (the bottleneck, default 100)
- per scale: `dense_relu_width` and `shared_latent_dim` (five values each; the
  shared-latent values are shared with the text side)
- text: `seq_len`, `embed_dim`, and the transformer schedule (number of blocks, width
  per block, heads, sublayers)

`EncoderEncoderPCN.__init__` builds the whole graph from this config. Wiring logic
(next_layers, the tap points, the `share_state_layer=` pairing, the compiled-sweep
attributes) is unchanged in behavior, only its widths and shapes come from the config.

Ship two configs:

- `NATIVE_7B`: the current literal widths and 572px resolution. Must reproduce today's
  model byte-for-byte.
- `COCO64_156M`: 64px, resized widths (Section 2), targeting ~156M params.

The default config keeps `NATIVE_7B` behavior so nothing else in the repo (the golden
gate, the compiled methods, `train_step`, generation) changes contract.

## Section 2: the 64px width scheme

Keep the conv channels and depth. At 64px the five tap feature maps become 262144,
131072, 65536, 32768, 16384 elements (conv2, conv4, conv6, conv8, conv9). The
`flatten -> inter(100)` layers therefore total about 51M parameters and this is forced
by the feature-map sizes, not tunable without changing the bottleneck or adding
pooling.

The tunable widths are the five `dense_relu_width` and five `shared_latent_dim`
values. Scheme:

- Size the shared-latent dims as a compression of each scale's feature map, keeping
  the 572px ordering (larger latent for the shallow high-resolution taps, smaller for
  the deep taps).
- Size the dense_relu widths in the same order and rough magnitude as the shared dims.
- Shrink the text transformer ladder sharply (a small `embed_dim`, few blocks, no or
  mild width growth), since the 512-to-4096 over 17 blocks alone is multi-billion.
- Set the text `seq_len` to 32 and `embed_dim` to 512 as defaults (COCO captions are
  short; the exact tokenizer and what fills the vectors is a Phase 4 data decision).

Concrete initial `COCO64_156M` estimate (the plan builds it, counts, and tunes):

- image conv stack unchanged (~15M), `flatten -> inter(100)` forced (~51M)
- shared_latent_dims by scale (conv2, conv4, conv6, conv8, conv9): 32768, 16384, 8192,
  4096, 2048 (a ~8x compression of each 64px feature map, keeping the 572px ordering)
- dense_relu widths by the same scale order: 32768, 16384, 8192, 4096, 2048
- image tunable dense (200*relu + 100*shared per scale) ~19M, so image side ~85M
- text: `seq_len=32`, `embed_dim=512`, a small transformer stack (about 6 blocks at
  width 512, 8 heads, no width growth) plus the five text taps mirroring the shared
  dims, estimated ~75M

That estimate totals roughly 160M. The implementation's first task builds it, counts
parameters exactly (the transformer param formula is verified there, not hand-derived),
and adjusts the config (most cheaply the transformer block count/width) to land near
156M. Landing within roughly 20 percent (about 125M to 190M) is acceptable for this
milestone.

## Section 3: conv depth at 64px

Keep the four maxpools. At 64px that yields a 4x4x1024 deepest feature map, the
standard VGG-on-small-images regime, fine for the deepest tap. Depth stays a config
field so a future variant can change it, but the `COCO64_156M` default keeps the
current conv structure so the retarget is a resize.

## Section 4: validation

Two levels, because the two configs have different validation semantics.

- `NATIVE_7B` must `GATE_MATCH` the existing 572px golden (`golden_baseline.npz` via
  `tools/rewrite_gate.py` and `tools/gate_compare.py`, relative tol 1e-4). This proves
  the config refactor did not change the known-good model. This is the primary safety
  gate for Section 1.
- `COCO64_156M` has no prior golden, so it is validated by: instantiates without error;
  total parameter count lands near the target (roughly 125M to 190M); one `pass_through`
  plus one relaxed weight step run batched on the H200 with all layer states finite (no
  NaN/Inf); the five shared-latent pairs are genuinely aliased
  (`dense4.state is dense2.state` and the other four); and both generation directions
  (`test_step(predict='img')` and `predict='txt'`) run and return finite tensors.

All runs use the existing cluster helper `tools/clusterrun.sh` on the H200.

## Risks and unknowns

- The config refactor touches the whole ~330-line constructor. The `NATIVE_7B`
  GATE_MATCH is the guard against silent breakage; if it fails, the refactor changed
  the architecture and must be fixed before the 64px config is trusted.
- PC stability at the new widths is unknown. The 64px model is new; the LR and
  divergence lessons from the functional runs may or may not transfer. Section 4 checks
  finiteness for a step, not long-run stability (that is Phase 4).
- Hitting exactly 156M by hand is not attempted; the plan tunes empirically.
- The text pathway shrink is a real architecture change (not just a resize); its taps
  must keep the shared-latent dims matched to the image side.

## Out of scope (Phase 4 and later)

Training on COCO-64, the caption data pipeline and tokenizer, tuning generation
quality, long-run PC stability at 64px, compiling the generation path, and the
batch-equivalence re-check at 64px (the I2 forward-GEMM artifact should shrink there).
Each is Phase 4 work with its own plan.
