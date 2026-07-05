# RUN_UNIF -- F_unif, the constructive repair test (8k, 150ep, pre-registered)

The paper's diagnosis says the PC energy F fails because its cross term is pure alignment with no
repulsion (established three ways: the PC arms, pinned BPonF under backprop, deeper relaxation driving
alignment higher). This run tests the implied repair: add a Wang-Isola uniformity term the relaxation
can consume, and ask whether held-out coupling appears.

## Design

Runner: `run_coupling_unif.py` (scratch variant of `run_coupling_scale.py`, byte-matched semantics,
lambda=0 reproduces the banked arm A trace digit for digit).

Energy, per example i, z_i = L2-normalized code_of(S_i) (the NS*CODE=64-dim code both decoders read;
chosen over latents() because encoder taps are constants during relaxation, so a uniformity on them
would have zero gradient into the relax loop and the question would be vacuous):

    F_unif(S_i) = F(S_i) + UNIF_LAMBDA * log mean_{j!=i} exp(-UNIF_T * ||z_i - z_j||^2)

Relaxation treats the other examples' codes z_j as constants (stop_gradient; the startup probe prints
that z_j receives exactly zero relaxation gradient). The weight step evaluates the same energy at the
relaxed, detached states with all rows live. The one deviation from the banked recipe: strict
batch-invariance of the relaxation no longer holds (negatives come from the batch); BATCHJ=128 was held
fixed for BOTH arms so the negative pool is consistent.

Arms, all at the banked 8k config (8000 train / 2000 held-out, 150 epochs, train2017, 156.7M params),
UNIF_LAMBDA=1.0, UNIF_T=2.0:

- PC-unif: the PC rule (relax-then-step, LARS, lr 2e-2), seeds 0,1,2. No LR retry was needed.
- BP-unif: backprop (Adam 1e-4) through the unrolled 8-step relaxation on F_unif (the free-latent
  pattern), seed 0. The ceiling arm.

Pre-registered rules: (1) repair works = held-out latent hits > 3/2000 on >= 2 of 3 PC seeds;
(2) necessary but not sufficient = PC drives held-out encoder uniformity out of the F-family band
(mean of unif_img/unif_txt < -1.0; band measured in E4: -0.01 to -0.52, InfoNCE systems at -3.8) but
retrieval stays at chance; (3) rule clause second instance = PC fails to move uniformity while BP
succeeds; (4) repair refuted = BP-unif also fails to transfer. Selection used train-side criteria only;
held-out retrieval was evaluated once per configuration.

## Results (every number from coupling_unif_results.json)

| arm | seed | move | TRAIN lat_retr | held-out lat hits (bar >3/2000) | held-out unif_img/txt (mean) | unif_code | align_cos | recon vs base | diversity | wall |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| PC-unif | 0 | 219% | 0.977 | 1 (+0.0 sigma) | -2.11/-2.22 (-2.16) | -3.57 | 0.412 | 0.0288 beats 0.0677 | 0.295 | 65m |
| PC-unif | 1 | 231% | 0.917 | 2 (+1.0 sigma) | -2.84/-2.18 (-2.51) | -3.78 | 0.142 | 0.0277 beats 0.0676 | 0.364 | 65m |
| PC-unif | 2 | 234% | 0.990 | 2 (+1.0 sigma) | -2.36/-2.47 (-2.41) | -3.66 | 0.351 | 0.0279 beats 0.0688 | 0.301 | 65m |
| BP-unif | 0 | 37%  | 0.0007 | 0 (-1.0 sigma) | -0.20/-0.33 (-0.26) | -1.37 | 0.225 | 0.1759 FAILS 0.0677 | 0.026 | 54m |

Train-side term trace: PC arms drive the batch uniformity u from about -2.4 at the start of training to
-3.9 (the near-uniform value for a 128-batch in 64 dims) and hold it there; the BP arm lets u rise from
-2.45 to -1.42 (it does not hold the term), fails to fit train (best 0.002), and its reconstruction
fails the train-mean baseline.

## Verdict (pre-registered branch 2, with rule 4 true as a clause)

Rules: [1 repair works] = False (0/3 crossed). [2 necessary but not sufficient] = True (3/3 optimized,
all at chance held-out). [3 rule clause second instance] = False (the local rule consumed the term;
BP did not, the inversion of that branch). [4 repair refuted, as clause] = True (BP-unif 0/2000).

BRANCH 2: NECESSARY BUT NOT SUFFICIENT, with a sharper interior than pre-registration anticipated:

1. The local PC rule CAN consume a repulsive energy term. Relaxation spreads the code (u to -3.9), the
   weight step propagates the spread into the encoders (held-out encoder uniformity leaves the F-family
   band, -2.2 to -2.8), and mean-collapse alignment breaks (align_cos 0.84 to 0.98 down to 0.14 to
   0.41). This kills the "the rule cannot consume repulsion" alternative before it is raised.
2. With repulsion, PC fits train coupling for the first time anywhere in the F family: train lat_retr
   0.92 to 0.99 (the banked F arms never left chance even in-sample). The failure boundary moves from
   "no per-pair coupling at all" to "per-pair coupling that memorizes and does not transfer" (held-out
   1, 2, 2 hits out of 2000, bar > 3).
3. The ceiling arm corroborates: backprop through the unrolled F_unif finds no transferable optimum
   either, and unlike free-latent BPonF on plain F it cannot even fit train or hold the term.

The sentence the paper gains: "Augmenting F with the missing repulsive term is consumable by the local
rule and repairs the geometry, restoring in-sample per-pair coupling (train retrieval 0.92 to 0.99 vs
chance-level under F), yet held-out matching stays at chance on every seed (1, 2, 2 of 2000 vs bar > 3)
and backprop through the same augmented energy transfers nothing (0 of 2000), so the transfer failure
is a property of the objective family, not of the missing repulsion alone nor of the local learning
rule."

## Notes

- BP-unif ran on an L4 (the cluster's full A100 was held by another user's job); peak 7.5 GB at
  BATCHJ=128, so the batch size held at 128 for both arms as pre-registered.
- Supplementary lambda=0.3 rung (PC, seed 0) replicates the branch-2 pattern at one-third the weight,
  so the verdict is not a lambda=1.0 artifact: move 256%, u to -3.88, TRAIN lat_retr 0.945, held-out
  2/2000 (chance), held-out unif_img/txt -1.96/-1.87 (out of the F-family band, dose-responsively less
  spread than lambda=1.0's -2.2 to -2.8), recon 0.0225 beats 0.0677, diversity 0.391. Not load-bearing
  for the verdict above.
- Reproduce: `sbatch -p normal -c 8 -J unif_pc_s0 --gres=gpu:L4:1 --mem=48G -t 1-00:00:00
  --export=ALL,RUNS1_SEED=0,UNIF_ARM=pc,UNIF_LAMBDA=1.0,UNIF_T=2.0 ~/hpc/unif_job.sh` (seeds 1,2 and
  UNIF_ARM=bp analogous). Raw per-job records in unif_unif_*.json, merged in
  coupling_unif_results.json.
