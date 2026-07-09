# Architecture

## Core modules
- `encoder_encoder_pcn.py`, the bidirectional PCN model. Builds the image and text branches from a PCNConfig and couples them through 5 shared-latent dense layers (share_state_layer at dense2/4, dense6/8, dense10/12, dense14/16, dense18/20).
- `pcn_config.py`, config definitions including NATIVE, COCO64_GEN, COCO64_156M.
- `train_coco64.py`, training entry that consumes a config.
- `conv_pcn_layer.py`, convolutional PCN layer, supports stride for bidirectional downsampling.
- `dense_pcn_layer.py`, dense PCN layer holding weights, bias, and state for relaxation plus the local LARS weight step, with an optional share_state_layer to couple two branches at a shared latent.
- `transformer_pcn_layer.py`, the text-path transformer building blocks (attention, add-normalize, positional encoding, and the per-group transformer wrapper).

## Experiment drivers
The `run_*.py` scripts each drive one experiment family. Current ones include `run_capacity_probe.py` and `run_coupling_capacity.py` (the capacity-ladder Phase-0 probes and driver, see `CAPACITY.md`), `run_coupling_scale.py` and `run_coupling_unif.py` (coupling and uniformity checks), `run_E1_bp_clip_baseline.py` and `run_E1_lars_infonce.py` (the E1 baselines), `run_step1_coco_gate.py`, `run_step1_coco_heldout.py`, and `run_step2_coco_dissociation.py` (COCO gate and dissociation), and `run_BPonF.py` / `run_BPonF_freelatent.py` (BPonF baselines).

## Downsampling paths (invariant-critical)
- NATIVE uses maxpool and must stay byte-identical.
- COCO64_GEN uses a stride-2 bidirectional conv (conv2d down, transpose-conv up, shared weights) so generative drive flows top-down.

## Why the invariants exist
- Byte-identical NATIVE is the control that every gated comparison depends on.
- The 28.7 GiB size is intentional for the capacity ladder, so OOM means a bug, not a reason to shrink.
