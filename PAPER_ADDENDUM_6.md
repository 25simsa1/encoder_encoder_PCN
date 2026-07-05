# ADDENDUM 6 integration text -- the F_unif repair test (branch 2 of the pre-registration)

Drop-in prose for the rewrite. Anchors name the rewrite's structure (the factorial subsection, claim 3a,
claim 4, the protocol section, the abstract). Every number below was recomputed from the committed JSONs
before writing: coupling_unif_results.json (five records: PC lambda=1.0 seeds 0/1/2, PC lambda=0.3 seed 0,
BP lambda=1.0 seed 0), BPonF_freelatent_results.json, E1L_results.json, res_8k*/res_20k* (banked arms),
analysis_latent_geometry.md (bands). Design and per-run detail in RUN_UNIF.md.

---

## 1. New Results subsection, placed after the factorial

### The constructive repair test: a repulsion-augmented energy the relaxation can consume

> The diagnosis says the cross term of F is alignment-only. The constructive test is to add the missing
> term and ask what appears. We add a Wang-Isola uniformity term to F over the 64-dimensional decoder
> codes, with the other examples' codes stop-gradiented during each example's relaxation so the energy
> stays per-example and the relaxation itself can consume the term; at lambda equal to zero the
> implementation reproduces the banked arm digit for digit.
>
> At eight thousand pairs and one hundred fifty epochs, matched to the banked runs, the PC rule on F_unif
> fits train coupling for the first time anywhere in the F family: train latent retrieval 0.977, 0.917,
> 0.990 on seeds 0, 1, 2, where the banked F arms never leave chance in-sample (0.000 to 0.008). It
> drives the batch uniformity of the codes to -3.9, near the uniform value for a 128-batch in 64
> dimensions, and the held-out encoder uniformity to -2.16, -2.51, -2.41, out of the F-family band
> (-0.01 to -0.52) and toward the InfoNCE systems (-3.7 to -3.8). It breaks alignment collapse, with
> held-out matched-pair cosine at 0.412, 0.142, 0.351 against the F-family 0.84 to 0.98, and
> reconstruction still fits (0.028 against the train-mean baseline 0.068). Yet held-out latent hits are
> 1, 2, and 2 per 2000, all below the pre-registered bar of more than 3. A lambda equal to 0.3 rung
> replicates the pattern dose-responsively (train 0.945, held-out 2 per 2000, encoder uniformity -1.96
> and -1.87). Backprop through the unrolled relaxation on the same F_unif at the matched configuration
> (Adam at ten to the minus four, one seed) did not hold the term (batch uniformity drifts from -2.45 to
> -1.42), did not fit train (0.0007), and its reconstruction fails the train-mean baseline (0.176
> against 0.068).

Result table, if the rewrite prefers tabular form (8k, 150ep, held-out pool 2000, bar more than 3 hits):

| arm | seed | train lat retr | batch uniformity u | held-out unif img/txt (mean) | held-out align | recon vs base 0.068 | held-out hits |
|:--|:--|--:|--:|:--|--:|:--|--:|
| PC-unif lambda 1.0 | 0 | 0.977 | -3.88 | -2.11/-2.22 (-2.16) | 0.412 | 0.0288 beats | 1 |
| PC-unif lambda 1.0 | 1 | 0.917 | -3.90 | -2.84/-2.18 (-2.51) | 0.142 | 0.0277 beats | 2 |
| PC-unif lambda 1.0 | 2 | 0.990 | -3.90 | -2.36/-2.47 (-2.41) | 0.351 | 0.0279 beats | 2 |
| PC-unif lambda 0.3 | 0 | 0.945 | -3.88 | -1.96/-1.87 (-1.92) | 0.494 | 0.0225 beats | 2 |
| BP-unif lambda 1.0 (Adam 1e-4) | 0 | 0.0007 | -1.42 (from -2.45) | -0.20/-0.33 (-0.26) | 0.225 | 0.1759 fails | 0 |

## 2. Claim rewordings

**(a) Claim 3a rescopes to objective-conditional.** Replacement sentence:

> The local rule's in-sample optimization deficit is objective-conditional, not universal. On the
> alignment-only F, backprop through the unrolled relaxation fits in-sample coupling (0.961 to 0.993)
> where the local rule sits at chance (0.000 to 0.008); on the repulsion-augmented F_unif the roles
> invert, and the local rule fits (0.917 to 0.990) where backprop through the identical unrolled
> relaxation does not (0.0007) at the matched configuration.

The T-sweep and free-latent sentences remain true for F and must now be scoped to F explicitly, for
example "on F, deeper relaxation budgets leave in-sample coupling at chance" and "on F, backprop through
the unrolled relaxation fits train on every seed."

**(b) The paper's sharpest sentence becomes the three-rung dissociation.**

> Adding the missing uniformity term repairs the latent geometry, and under the local rule repairs
> in-sample coupling (train retrieval 0.92 to 0.99), but held-out transfer appears at neither rung
> (1, 2, 2 hits per 2000 against a bar of more than 3), while direct contrastive training of the same
> encoders at the same scale transfers robustly (7, 4, 5 per 2000 at eight thousand pairs; 13, 11, 13 at
> twenty thousand). The transfer gap is therefore not explained by geometry alone.

**(c) Claim 4 gains its fourth instance, in reverse.**

> Removing the alignment monopoly does not remove the transfer failure. With the uniformity term
> consumed (batch uniformity at -3.9, matched-pair alignment down to 0.14 to 0.41), per-pair transfer
> still does not appear, so mean-collapse alignment is a symptom of F, and curing the symptom does not
> cure the transfer failure.

## 3. Hedges, non-negotiable

- BP-unif is one configuration and one seed. Write "did not hold the term at the matched configuration
  (Adam at ten to the minus four, one seed)". Never a general inability claim about backprop on F_unif.
- Pre-registered rule 3 fired in inversion. Report as a surprise, not a planned finding. Suggested
  wording: "Pre-registration anticipated the local rule might fail to consume a repulsive term that
  backprop could; the data returned the inversion, which we report as a surprise."
- The one recipe deviation: the uniformity negatives come from the training batch, so strict
  batch-invariance of the relaxation no longer holds for this arm family; the joint batch was held at
  128 for both arms so the negative pool is consistent.

## 4. Discussion and future work

> What InfoNCE has that the fitted F_unif lacks is now the precise open question. The candidates are
> (i) direct encoder-shaping, with no generative pathway and no relaxation in the loop between the
> objective and the encoder weights; (ii) the softmax hard-negative structure of InfoNCE against the
> Gaussian-potential form of the uniformity term; and (iii) the decoders' gradient traffic, which under
> F_unif shares the weight step that must also propagate the repulsion. We do not adjudicate among these
> here.

Protocol section gains one sentence (an anticipated reviewer question for any repair experiment):

> For the repair experiment, configuration selection used train-side criteria only, whether the arm
> optimizes the uniformity term and whether it is stable; held-out retrieval was evaluated once per
> final configuration and was never a selection criterion.

## 5. Page 1

The headline framing stays diagnosis-shaped and the repair test is its strongest interior evidence. One
sentence in the abstract:

> Adding the missing repulsive term repairs the latent geometry and, under the local rule, in-sample
> coupling, but not held-out transfer.

Do not retitle around the repair; branch 2 is not the repair-works branch.

## 6. Sources for every number

| number | source |
|:--|:--|
| PC-unif train 0.977/0.917/0.990, u to -3.88/-3.90/-3.90, held-out unif means -2.16/-2.51/-2.41, align 0.412/0.142/0.351, hits 1/2/2, recon 0.0277-0.0288 vs base 0.0676-0.0688 | coupling_unif_results.json (arm pc, lam 1.0) |
| lambda 0.3 rung: train 0.945, hits 2/2000, unif -1.96/-1.87 | coupling_unif_results.json (arm pc, lam 0.3) |
| BP-unif: train 0.0007 (best 0.002), u -2.45 to -1.42, hits 0/2000, recon 0.1759 vs 0.0677 | coupling_unif_results.json (arm bp) |
| banked PC-on-F in-sample chance range 0.000 to 0.008 | res_8k*.json, res_20k*.json train lat_retr across arms |
| free-latent BP-on-F fits 0.961/0.989/0.993 (held-out 0/2/6) | BPonF_freelatent_results.json (lr 1e-4) |
| contrastive transfer 7/4/5 at 8k, 13/11/13 at 20k | E1L_results.json |
| F-family uniformity band -0.01 to -0.52, InfoNCE -3.7 to -3.8, F-family align 0.84 to 0.98 | analysis_latent_geometry.md, banked res_*.json |
| design, stop-gradient probe, lambda=0 equivalence, verdict | RUN_UNIF.md, run_coupling_unif.py, commits c69edfd/77db885/68e7a64 |
