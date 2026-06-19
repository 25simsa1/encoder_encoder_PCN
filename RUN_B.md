# Run B-prime — plain-LARS scale push (330M → 3B): where does from-scratch fail?

Conference-path Run B-prime. Run A showed muP UNDER-moves at scale and plain LARS + the transferred
LR is the lever (330M reached "useful" generation from scratch). Run B-prime tests the simplest
hypothesis before building any InfoNCE warm-up: **does plain LARS + the transferred lr=2e-2 keep the
model generating from scratch as size grows past 330M, and is there a failure scale?**

Ran `run_B_scale_push.py` (committed 2dec932) on a RunPod **A100-SXM4-80GB**. Recipe identical to the
validated one (single energy F, GELU, LARS + bias trust floor, relax-then-step, dense multi-scale
anchors, A_GEN≥A_cross, all grads via GradientTape), **PLAIN LARS / standard parameterization** (uniform
LR, fan-in init, tiny DEC_SD decoder init — NO muP). Data MNIST, N=64 distinct images each with a
distinct random caption, chance retrieval 0.016. Fixed budget **1500 steps**, lr **2e-2** (the Run-A
transferred LR), local-disk checkpoints. Per-size guard catches divergence AND OOM and continues the
sweep.

## Results (from scratch, plain LARS, lr=2e-2, 1500 steps)

| size | params | weight movement | text→image retrieval (chance 0.016) | diversity | max\|w\| | diverged? | OOM? | band | peak GPU | wall-time |
|:--|--:|--:|--:|--:|--:|:--:|:--:|:--|--:|--:|
| 330M  | 330.2M  | 71.8% | 0.422 (≈27×) | 0.413 | 8.52 | no | no | **useful** | 4.0 GB  | 424s |
| 704M  | 703.8M  | 64.7% | 0.281 (≈18×) | 0.521 | 2.58 | no | no | **useful** | 9.9 GB  | 446s |
| 1.49B | 1494.3M | 62.7% | 0.297 (≈19×) | 0.352 | 2.76 | no | no | **useful** | 17.9 GB | 621s |
| 2.99B | 2992.7M | 48.5% | 0.422 (≈27×) | 0.347 | 1.79 | no | no | **useful** | 42.1 GB | 629s |

- **Largest size that fit + ran in 80GB: 2.99B** (peak 42.1 GB — roughly half the card, so ~5–6B would
  still fit; the limit was not reached).
- **No size OOM'd, no size diverged.** max|w| stayed bounded (1.8–8.5, far below the 1e3 guard).
- Wall-time ~420–630s per 1500-step run; it grows only mildly with size (batch-1 is largely
  kernel-launch-bound). Full 4-size sweep ~37 min.

## Failure scale

**None found in the swept range.** Movement declines gently with width (71.8 → 64.7 → 62.7 → 48.5%) but
stays in the **useful** band (≥48%) all the way to 3B, and **retrieval never falls toward chance** — it
holds at 0.28–0.42 (≈18–27× chance) at every size, including 3B (0.422). So plain-LARS from-scratch does
**not** drop below the 22–48% collapse-break band and does **not** lose generation anywhere up to 3B.

## Verdict (blunt)

**Plain LARS keeps generating as size grows — there is no from-scratch failure scale up to 3B.** With
the transferred lr=2e-2 and standard parameterization, the model trains to "useful" generation from
scratch at 330M, 704M, 1.49B, and 2.99B alike, in a fixed 1500-step budget, with no divergence and no
OOM. At-scale from-scratch generation is therefore **trivially in hand** with plain LARS; **an InfoNCE
warm-up is NOT needed in the affordable range (≤3B).**

The warm-up-relevant scale — if one exists — is **beyond 3B**. Since 3B used only 42 GB peak, the
natural next probe is to push further (≈5–6B fits 80GB; larger needs multi-GPU or gradient sharding) to
find where, if anywhere, plain LARS finally fails. Until then, the simplest recipe (plain LARS +
transferred LR) is the at-scale control, and the warm-up work can wait.

Honest scope: this is the MNIST / N=64 distinct-caption regime at a fixed 1500-step budget; "useful"
means varying, above-chance-recognizable generation (retrieval 18–27× chance), not crisp samples. It
establishes that the optimization (weight movement + generation) does not break with width up to 3B —
the question Run B-prime was built to answer.

Artifacts: `runB_results.json`, `run_B_scale_push.py` (the script, 2dec932).
