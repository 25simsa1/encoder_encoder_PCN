# Design: the isometry constraint, orthogonal weight geometry so transpose approximates inverse

Date: 2026-07-13

## Goal

Give the shared weights a geometry where each edge's top-down transpose is automatically a
right-inverse of its bottom-up map (`WᵀW ≈ I` per edge), by adding a LOCAL soft
orthogonalization term to the weight update during ordinary recon training. This is a
CONSTRAINT shaping the geometry, not a second objective, so it never enters the tug-of-war
over the shared weights that closed all three calibration families. Strictly bidirectional PC.

## Background (why this is categorically different)

Three training families (end-to-end CHL contrast, expressing-free-phase calibration, per-edge
cascade-consistency) showed the identical empty window, gentle pressure is out-anchored by the
recon interleave and moves nothing, recon-equal pressure rips the shared weights off the recon
equilibrium instantly. The root is that every one of them was a SECOND OBJECTIVE fighting recon
over the same parameters. The isometry constraint has no target and no error signal, it is a
weight-space regularizer computed from the layer's own matrix alone. And the analysis of the PC
energy showed the two error directions of each edge are SIMULTANEOUSLY satisfiable exactly when
`WᵀW = I` on the signal subspace (the top-down preimage `s Wᵀ` then reproduces `s` bottom-up),
so with orthonormal columns the joint relaxation minimum carries the top-down content instead of
compromising it away.

## The mechanism

- Per-layer plain float `iso_eta` (default 0.0, byte-identical off). At the END of
  `update_wts`, after the normal local step, when on,
  - Dense (`wts` is (in, out), in >= out on this image path),
    `wts += iso_eta * wts @ (I_out - wtsᵀ wts)`,
    the classical local orthogonalization flow, fixed point `wtsᵀwts = I` (orthonormal columns,
    unit norms, so it is also a norm stabilizer and replaces weight-norm for these runs).
  - Conv (`wts` is (kh, kw, in, out)), the standard kernel-orthogonality surrogate, reshape to
    K of shape (kh*kw*in, out), apply the same flow, reshape back.
- Locality, the term uses only the layer's own weight matrix, no error signal, no other layer,
  no backprop. PC-legal trivially, and it does not compete with the recon gradient, it projects
  the geometry as recon learns.
- `--isometry ETA` in train_coco64.py sets `iso_eta` on the image-path conv/dense layers
  (including the latent dense layers' weights, they are edges too). Recon training otherwise
  uses the stable recipe (lr 1e-3, wd 3e-2, clip 400, gelu), WITHOUT --weight-norm (the
  constraint bounds norms itself and the two reparameterizations would interact).

## The run

Train COCO64_GEN recon FROM SCRATCH with `--isometry` (warm-starting is wrong here, the
recon-trained scales are far from orthonormal and snapping them would destroy the equilibrium;
from scratch the geometry and the recon co-develop). Sweep eta lightly if needed (1e-3 first).
Then the standard retest, latent_source_diag on the result at the pi schedule (0, 0.25), the
PLAIN 1:1 schedule at an adequate rate (with `WᵀW ~ I` the plain joint minimum is the newly
reachable one), and the boost baseline.

## Validation

- Unit tests, `iso_eta = 0` is a no-op (byte-identical), the flow decreases `||WᵀW - I||` on a
  random matrix, and column norms approach 1.
- COCO64 gate vs the banked reference (layer files change, defaults must stay byte-identical).
- Training telemetry, recon energy must still reach a usable floor under the constraint (the
  constraint costs capacity, watch the floor), plus the standard stability greps.
- The decisive retest as above, does decode(img-set) carry identity under the plain or pi
  schedule once the geometry is orthonormal.

## Risks

- The constraint costs representational freedom, recon may not reach a usable floor (watch the
  energy floor against the ~0.006 unconstrained reference).
- GELU and the strided downsampling are not inverted by a transpose even with orthonormal
  kernels, the relaxation must absorb the mismatch, this is the main residual risk.
- Kernel orthogonality is a surrogate for true conv isometry, adequate for a probe.
- eta too large fights the recon gradient after all (a projection applied too hard each step),
  eta too small never reaches the orthonormal manifold, one light sweep.

## Out of scope

Fix B (text alignment, only after a decode milestone), weight-norm composition, held-out, the
ladder. This is the LAST mechanism-level generation swing, if it nulls, the chapter banks.
