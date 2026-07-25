# PCMAX binding diagnosis and where the wall is (design note for the lead)

Written 2026-07-25 by the paper-side collaborator after joining the PCMAX search. This consolidates
why the real PCMAX arm does not fit the coupling, backed by the driver code and the run diagnostics,
and lays out the one hard sub-problem the sweep keeps hitting. It is a synthesis for a decision, not a
launch order. Evidence lives in LOG 2026-07-25 (three entries).

## The question this answers

Does real bidirectional block-local PC fail to fit the cross-modal coupling because local credit
assignment fundamentally cannot bind (H2, the thesis), or because the delivery of the coupling signal
is under-powered or mis-tuned (H1, fixable)? The distinction gates whether the paper writes PCMAX as a
hardened negative or keeps tuning.

## Root cause, from the code

In `weight_step_local` (run_pcmax_capacity.py l511) the joint objective is
`L = F_pcmax + jointw * infonce(taps_from_states(Zi,Zt))`. Both the InfoNCE term and the `cross` term
inside `F_pcmax` read the taps through `taps_from_states`, which applies only the tap projection
weights (Wi*, Wt*) to the relaxed states. So the coupling gradient reaches only the tap projections.
The interior encoder blocks (convs c*, attention Wq/Wk/Wv/Wo, FFN f*) are updated purely by the
`e_bu` and `e_td` reconstruction energy. They learn to predict the relaxed states and nothing else.

Consequence. Cross-modal coupling can shape the interior encoder only if the relaxed states are
already coupling-shaped, and the sole mechanism that shapes them that way during inference is the HEP
highway. The highway is therefore the whole ballgame for interior binding.

## Why the highway cannot carry it

The calibration and the per-block RMS diagnostic show the highway has no stable operating point.

- At a deliverable scale (alpha giving highway/local ratio about 0.03) the per-step interior state
  motion is identical with the highway on versus off (RMS(dZ)/RMS(Z) 2.6e-2 versus 2.7e-2). The
  InfoNCE latent gradient is about 1e-6 the size of the local reconstruction update, so the injected
  nudge is numerically inert.
- Forcing the ratio up toward 0.3 and beyond destabilizes the relaxation. relaxF climbs into the 1e10
  range and the run diverges.

There is no alpha where the highway both moves the interior states and stays stable. Bmu, which is the
same energy and the same coupling delivered by backprop through the encoder, fits to 0.99 without
trouble. The failure is localized precisely to local delivery of a coupling signal that is subdominant
to reconstruction.

## The A_CROSS positive control and what it rules in

To test whether any local channel other than the highway can bind, I added a gated `PCMAX_ACROSS`
knob (default 1.0, byte-identical, parity safe) and cranked the energy's own cross term with the
highway off. The cross term pulls the shared latent toward both taps during inference, a PC-native
coupling channel that needs no highway.

Result at A_CROSS=10, seed 0, 156M/20k. align climbs 0.001, 0.22, 1.000 by epoch 4 while train
retrieval stays at zero, then the energy grows toward divergence. This is representational collapse.
Every embedding rotates to one direction, which gives perfect global alignment and zero per-pair
discriminability. It is the paper's "alignment without binding" reproduced inside the PCMAX arm.
A_CROSS=100 is queued for confirmation.

The honest caveat. The cross term is a regression coupling with no contrastive negatives, and
regression-only coupling is known to collapse. So this probe does not by itself prove H2. It shows
that the collapse-free coupling channel is the InfoNCE, and the InfoNCE reaches the interior encoder
only through the highway.

## The wall, stated cleanly

Two coupling channels, two failure modes.

1. Contrastive InfoNCE, which avoids collapse, reaches the interior encoder only via the highway,
   which is inert at deliverable scale and divergent when forced.
2. Regression cross-consistency, which does reach the interior states through inference, collapses
   because it has no negatives.

No local channel we have delivers collapse-avoiding coupling into the interior encoder in a stable
regime. That is a sharper and more defensible statement than the old arm-B story, and it is very much
on-thesis. It is not yet a single-experiment proof, because the clean H1 test has not been run.

## The one clean experiment that would settle H1 versus H2

Deliver contrastive coupling, negatives included, locally to the interior encoder, and ask whether it
can bind without either collapsing or diverging. Candidate routes, for the lead to weigh since this is
architecture territory.

- Give the interior blocks their own local contrastive target rather than routing the boundary
  InfoNCE gradient through fixed random highways. A per-block InfoNCE on the block outputs against a
  detached partner, so negatives are present at every depth.
- Normalize the highway injection against the local update magnitude per block (a stronger form of the
  ANORM path) so the contrastive nudge is neither 1e-6 nor destabilizing, holding a fixed fraction of
  the local step at every depth and every training stage.
- A closed-form or teacher-forced positive control. Clamp the interior states to a known-coupled
  solution taken from the fitted Bmu checkpoint and run only the block-local weight update. If the
  encoder becomes discriminative, the local update can bind and the problem is delivery. If it cannot,
  the local update itself cannot propagate binding. This is the cheapest and most decisive, and it
  mirrors the ridge_td fixed-point move the project used to settle the generation ceiling.

I recommend the teacher-forced control first. It is cheap, needs no new stable training regime, and it
cleanly separates delivery from the fundamental question.

## What it means for the paper either way

- If a local contrastive delivery binds, PCMAX becomes the credible PC arm that fits, and the
  headline is whatever its transfer does at matched fit.
- If it cannot, the paper states the mechanism directly. At matched training capacity the local rule
  either fails to acquire the coupling (highway inert), destabilizes trying (highway forced), or
  collapses (regression coupling). Only credit assignment that carries the contrastive gradient
  through the encoder binds. That is a stronger and more mechanistic version of alignment without
  binding, and it is exactly the claim a PC-literate reviewer will find credible.
