# Architecture

## Core modules
- `encoder_encoder_pcn.py`, the bidirectional PCN model. Builds the image and text branches from a PCNConfig and couples them through 5 shared-latent dense layers (share_state_layer at dense2/4, dense6/8, dense10/12, dense14/16, dense18/20).
- `pcn_config.py`, config definitions including NATIVE, COCO64_GEN, COCO64_156M.
- `train_coco64.py`, training entry that consumes a config.
- `conv_pcn_layer.py`, convolutional PCN layer, supports stride for bidirectional downsampling.
- `dense_pcn_layer.py`, dense PCN layer holding weights, bias, and state for relaxation plus the local LARS weight step, with an optional share_state_layer to couple two branches at a shared latent.
- `transformer_pcn_layer.py`, the text-path transformer building blocks (attention, add-normalize, positional encoding, and the per-group transformer wrapper).

## Experiment drivers
The `run_*.py` scripts each drive one experiment family, capacity probes, coupling and uniformity checks, E1 baselines, COCO gate and dissociation, and BPonF baselines. The ones current to the active work are named in `docs/STATE.md`.

## Downsampling paths (invariant-critical)
- NATIVE uses maxpool and must stay byte-identical.
- COCO64_GEN uses a stride-2 bidirectional conv (conv2d down, transpose-conv up, shared weights) so generative drive flows top-down.

## Why the invariants exist
- Byte-identical NATIVE is the control that every gated comparison depends on.
- The 28.7 GiB size is intentional for the capacity ladder, so OOM means a bug, not a reason to shrink.
