# PC-max: scaling bidirectional PC with muPC + error highways (design)

Date 2026-07-24. Lead directive: the capacity ladder never scaled bidirectional PC the way the two
published scaling methods prescribe. This spec defines the "best-shot PC" arm that applies both,
validated at 156M/20k before any bigger rungs. Approved decisions: both methods combined; jobs
9514/9515 keep running as comparison points; weight updates fully per-block local (no cross-layer
backprop anywhere); the arm is bidirectional PC (reciprocal prediction edges); budget ~60-100 GPU-h
seed-0, optional +50-80 GPU-h seeds 1/2 extension only if warranted.

## Source methods

- muPC (Innocenti, Achour, Buckley, NeurIPS 2025; arXiv:2505.13124). Standard PCN parameterization
  is unscalable: the inference landscape ill-conditions with size/training and the forward pass
  vanishes/explodes with depth. Fix: Depth-muP reparameterization -- N(0,1) init with per-layer
  premultipliers (hidden x 1/sqrt(fan_in), readout x 1/fan_in), 1/sqrt(L)-scaled residual branches,
  T ~ number of hidden layers inference steps (one step degrades), zero-shot LR transfer across
  width and depth.
- HEP (Mohammadi & Ororbia 2026; arXiv:2606.22744). The PC learning signal decays O(lambda^(L-i))
  hopping back from the clamped boundary. Fix: fixed random highway matrices V_{L->i} (sigma_v=1e-3)
  inject the boundary error into hidden states at every inference step (strength alpha,
  stop-gradient), plus RMSNorm forward pass and Adam-on-states (stats reset per batch).

## Why the current ladder arm cannot host either method

`run_coupling_capacity.py` arm B gives free states ONLY to the 4 shared taps; `weight_step`
backprops the energy through the whole encoder forward, and warmup/jointw InfoNCE is plain backprop.
No interior states exist for muPC to condition or highways to reach. PC-max removes that backprop.

## The PC-max arm (new driver `experiments/run_pcmax_capacity.py`)

Fork of `run_coupling_capacity.py`, family discipline (byte-copy everything not deliberately
changed; every deviation listed in the header docstring). `run_FA_lars_infonce.py` is the template
for fixed-matrix discipline and digit-match gates.

1. muP parameterization. All weights drawn N(0,1) (same generator, same draw order as baseline so
   the parity gate can hold), used as (x@W)*MULT[k]: hidden 1/sqrt(fan_in); decoders/proj 1/fan_in.
   RMSNorm with learnable gains at block inputs (image blocks 2-4; text pre-attention + pre-FFN per
   block). Residual branches scaled RSCALE=1/sqrt(2*NBLK). Gains are new ckpt keys; `__pcmax=1`
   arch marker saved in the npz.
2. Bidirectional interior states. Free states: image Z1..Z4 (post-block), text Z0..Z3 (residual
   stream post-block), plus the 4 tap states S_k. Reciprocal edges: bottom-up f_l = the block
   forward; top-down g_l = untied weights (per the 2026-07-17 untied-td design rule): image
   conv2d_transpose stride-2 k3 (td_c1 predicts the clamped input image), text position-wise dense
   [DM,DM] for b=1..3 (block 0 sits on the discrete tokens; emb+pos fold into f_0 so they train).
   Energy F = sum_l mse(Z_l - f_l(Z_{l-1})) + BIDIR_W*sum_l mse(Z_{l-1} - g_l(Z_l))
   + A_CROSS*sum_k(mse(S_k-it_k)+mse(S_k-tt_k)) + A_GEN*(recon), taps computed FROM the states.
3. Inference. Feedforward init (bu errors exactly zero at t=0), T=PCMAX_T_INFER (default 16) steps
   of hand-rolled Adam-on-states (m,v reset per batch). Each step adds the highway term.
4. Highways. eps = -grad InfoNCE wrt the pre-norm concat latents (small tape over the tap outputs,
   O(B^2*sum DIMS) ~ 0.1% of a trunk forward, recomputed every step), split at DIMS boundaries.
   Routing: text seg b -> text block b; image seg0->Z2, seg1->Z3, seg2->Z4 (the block feeding that
   tap), seg3 (bottleneck) -> Z1 so every block has a highway. V matrices map the segment to the
   state's channel/DM axis and broadcast over positions (justified: taps are mean/flatten-pooled, so
   the tap error is position-uniform; full flattened-state maps are ~90 GB/block at 7.7B). Total
   ~1.7 GB fp32 at 7.7B. Discipline: dedicated Generator(seed+20011), fixed draw order, non-
   trainable, NOT in P, never saved. Tap states get no highway (the cross terms already deliver
   their error).
5. Local weight update. States enter the weight tape as constants, so each energy term touches
   exactly one block's weights; one tape over the summed energy is automatically block-local. The
   JOINTW*InfoNCE term is computed from taps of constant states, so its gradient reaches only the
   boundary-adjacent tap heads (Wi*, wbn, Wt* and biases) -- the PC output layer, nothing upstream.
   LARS trust ratio byte-copied. NO InfoNCE warmup phase for PC-max (backprop-free by
   construction); resume therefore stays exact (epoch-shuffle RNG only).
6. MUPC_ONLY arm ("Bmu"): muP parameterization on the verbatim arm-B code path (no states, no
   highways) -- the cheap ladder-hardening arm; ckpt `cap_Bmu_*` never collides with banked
   `cap_B_*`.

Env: inherited RUNS1_*/CAP_* unchanged; new PCMAX_ALPHA, PCMAX_T_INFER(16), PCMAX_SIGMA_V(1e-3),
PCMAX_STATE_LR(1e-2), PCMAX_STATE_OPT(adam|gd), PCMAX_BIDIR_W(1.0), PCMAX_PARITY(0), PCMAX_DIAG(1),
PCMAX_FITSTOP(0=off; set 0.99 for the budget lever). RUNS1_ARMS gains "PCMAX" and "Bmu".

## Gates (all pass on CPU before any GPU submission)

1. Parity: PCMAX_PARITY=1 (baseline init law, no gains/td/norm/RSCALE, arms A,B) must match the
   baseline driver smoke on every printed numeric line (FA transpose-gate standard).
2. CPU smoke (PCMAX arm, T=2, alpha>0): F trend non-increasing over inference steps (strict
   monotone under PCMAX_STATE_OPT=gd), no NaN/Inf, ckpt reloads with all 68 baseline keys at
   baseline shapes, verdict prints.
3. Signal-propagation diagnostic (HEP Fig-1 analog): per inference step, per-block
   RMS(dZ)/RMS(Z), alpha=0 vs alpha>0 on the same batch. Expected: alpha=0 deep blocks near-silent
   early; alpha>0 all blocks move from step 1. Also a paper figure.
4. Probe compatibility: `tools/mechanism_probe.py` gets a `__pcmax` branch (premultipliers
   reconstructed from shapes, gains from file); gate = probe latents vs driver latents
   max-abs-diff < 1e-5 on smoke examples.

## Phase-1 GPU campaign (Colby, <=100 GPU-h, submission on explicit go)

alpha x T probe at 156M/20k, short runs (~10-20 ep): alpha in {0, 0.01, 0.1, 1.0} x T in {8,16}
(alpha=0 is HEP's own control; HEP Table 4 anchors the range). Then full seed-0 runs: PC-max at the
best (alpha,T) with early-stop at fit (train lat_retr >= 0.99 -- epochs beyond fit do not improve
the verdict), and Bmu under the stability recipe (~13 h). Judgment: readout battery +
`tools/category_probe.py` vs banked BP (e1l 20k) and PC (cs_B 20k) ckpts on the same split.

Pre-registered branches: category lift >>1.3x at matched fit => the dissociation is a
signal-delivery artifact curable with local highways; ~1.3x at matched fit => the thesis is
hardened against "you never scaled PC with the published methods"; fails to fit => instability
edge recorded (one LR retry per LADDER_V2 convention). Seeds 1/2 only on an interesting seed-0.
Ladder rungs 330M+ under muP LR transfer are Phase 2, separate go.

## Invariants respected

NATIVE untouched (new driver only). The 8k >3/2000 bar unmoved. No capacity cuts on OOM
(CAP_RECOMPUTE instead). Known risks pinned in the driver docstring: block-level (not neuron-level)
locality of the transformer term; Adam-on-states transient F rises (trend gate); highway term
shrinking as InfoNCE trains (diagnostic monitors; RMS-normalized alpha is a pre-registered fallback
knob, never a silent change); decoder premultiplier vs DEC_SD changes early recon magnitude.
