# Run A — scale sweep (undertraining onset) + µP-vs-SP LR transfer

Conference-path Run A. Ran `run_A_scale_mup.py` (committed 3ae39ad) on a RunPod **A100 80GB**, GELU +
LARS + bias trust floor, relax-then-step, dense multi-scale anchors, A_GEN≥A_cross, single energy F,
all grads via GradientTape. Data MNIST, **N=64** distinct images each with a distinct random caption,
chance retrieval **0.016**. Fixed budget **1500 steps**, base LR 2e-2, local-disk checkpoints (/root).

Sizes (scaling DM / channels / shared-latent DIMS, depth fixed): **5.54M** (DM=128, DIMS=[768,768,512,512]),
**55.52M** (DM=408), **330.20M** (DM=996, DIMS=[5990,5990,3994,3994]). **330M fits in 80GB comfortably**
(~5GB used at batch-1). Wall-time ~321–345s per 1500-step run **regardless of size** — batch-1 is
kernel-launch-bound, not compute-bound — so size barely changes cost.

> Provenance note: the pod was terminated before `runA_results_mup.json` could be scp'd; the committed
> JSON is reconstructed verbatim from the streamed `runA_mup.log`. The separate SP scale-sweep run was
> stopped after its first size (its PART 2 is redundant — PART 2 of the µP run already ran *both* the µP
> and SP transfers under matched conditions, which is the headline).

## PART 1 — from-scratch scale sweep (µP, 1500 steps, lr=2e-2)

| size | params | weight movement | text→image retrieval (chance 0.016) | diversity | band |
|:--|--:|--:|--:|--:|:--|
| small  | 5.54M  | 155.3% | 0.109 (≈7×) | 0.290 | useful |
| medium | 55.52M | 56.3%  | 0.125 (≈8×) | 0.202 | useful |
| large  | 330.2M | 28.7%  | 0.078 (≈5×) | 0.189 | collapse-break |

Under µP the per-layer LR is shrunk by `m^-0.5` with width, so movement falls monotonically
(155→56→29%) and the 330M model lands at the collapse-break edge with retrieval starting to drop.
**No size cleanly undertrains** at this budget (the <22%-movement / <5×-chance-retrieval thresholds are
not crossed), and the SP transfer below shows 330M is actually fine (67% / 0.391) under standard
parameterization. **Undertraining onset is not reached in [5.5M, 330M] — go larger to find it.**

## PART 2 — µP-vs-SP LR transfer (the headline)

Small-model (5.54M) LR sweep, 500 steps: 5e-3→18.1%, 1e-2→29.5%, **2e-2→62.6% (useful, chosen)**,
4e-2→152%. **Tuned LR = 2e-2**, transferred unchanged to the **330M** model under each parameterization:

| 330M @ lr=2e-2 | weight movement | retrieval (chance 0.016) | diversity | diverged? | band |
|:--|--:|--:|--:|:--:|:--|
| **µP** | 27.6% | 0.125 (≈8×)  | 0.199 | **no** | collapse-break |
| **SP (plain LARS)** | **67.0%** | **0.391 (≈24×)** | 0.436 | **no** | **useful** |

## Verdict (blunt)

**Both µP and SP transfer the small-model LR to 330M without diverging — and plain LARS/SP transfers
*better*.** SP gives higher movement (67% vs 28%) and far higher retrieval (0.391 vs 0.125) at the same
LR. **µP does not earn its keep at ≤330M on this task**: its `m^-0.5` scaling deliberately shrinks the
large-model LR (×0.36) and ends up *under*-moving it. The lever is **LARS**, whose per-layer trust ratio
already makes relative weight movement width-stable; µP adds nothing here and slightly hurts.

**Implication for the next (more expensive) runs: lean on plain LARS/SP, not µP.** The 330M-under-SP
result (67% movement, retrieval 0.391 ≈ 24× chance, no divergence) is the strongest 330M generation in
the sweep and the natural operating point. To *find* the from-scratch undertraining onset we would need
to go beyond 330M (or cut the step budget); it is not reached at 330M / 1500 steps here.

Artifacts: `runA_results_mup.json` (reconstructed), `run_A_scale_mup.py` (the script, 3ae39ad).
