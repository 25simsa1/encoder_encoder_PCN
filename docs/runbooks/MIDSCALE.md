# Mid-scale bidirectional image–text PCN — positive control

**Question:** the 7.7B `encoder_encoder_PCN` produced mush. Was that *undertraining* (weights moved only
~1.4% at lr=5e-4 — too many params for the step budget), or an architectural limit? **Test:** build a
shrunk version of the *same* design, train it until the weights actually move (target ≥40%), and see
whether text→image / image→text generation works.

**Verdict (blunt): GENERATES — the full design generates when properly trained; the 7.7B mush is
consistent with undertraining, not an architectural limit.** With the weights moved 603% and both
encoders alive, text→image produces **varying, recognizable-above-chance** output (diversity 0.43,
retrieval 0.30 ≈ 19× chance), image→image reconstructs (MSE 0.030), and image→text beats baseline 8×.
**Important nuance (reported honestly):** getting there required avoiding a *second*, distinct failure
mode — a **dying-ReLU that killed the text encoder** under the aggressive learning rate. So "move the
weights" was necessary but, at this scale/lr, not sufficient on its own: unit death had to be prevented
too. Both failure modes are training pathologies; neither is the architecture being unable to generate.

Artifacts: `midscale_grid.png` (leaky run — generates), `midscale_grid_run1_relu.png` (relu run — dead
text encoder), `midscale_F.png`, `midscale_results.json`, `midscale_results_run1_relu.json`. Reproduce:
`python3 midscale.py` (CPU, ~10 min; `MID_LEAKY=0` reproduces the dead-encoder run).

---

## Model (same architecture as the 7.7B, shrunk) and recipe

- **Option-C bidirectional**, faithful to `encoder_encoder_pcn.py` but shrunk: conv image encoder
  (32→64→128 ch, 3 pool blocks) + 4-block transformer text encoder (d_model 256, 4 heads), each
  predicting into **NS=4 shared latents** (dims [4096,4096,2048,2048]) with **dense per-scale anchors**
  (every scale anchored by *both* an image tap and a text tap — the L3 lesson), and **decode-to-input
  for both modalities** (image decoder + text decoder via per-scale projection→concat→linear — the
  Option-C structure). **50.5M params.**
- **Validated recipe (not the layer classes' hand-written update math):** one scalar energy F; **all**
  updates via `tf.GradientTape`; **LARS with the +1e-3 bias trust floor**; **relax-then-step** (8 inner
  relaxation steps on the latents, then one weight step); true ReLU derivative via autodiff;
  **generative precision A_GEN=2.0 ≥ cross precision A_CROSS=1.0** (the L4 lesson).
- **Data:** MNIST, **64 distinct images, each with a distinct random caption** (8 tokens, vocab 32) — no
  class-label shortcut, so text→image must carry per-sample content. *MNIST (not CIFAR/COCO) chosen
  deliberately: the question is a clean yes/no on recognizable, varying generation within a CPU
  time-box; MNIST gives an unambiguous recognizability read-out, CIFAR fidelity at this budget would be
  hard to call. The question is state-of-evidence, not image fidelity.*
- **LR pre-check** (per instruction — largest lr that moves weights without diverging): lr ∈ {1e-3, 5e-3,
  1e-2, 2e-2} all finite/stable at 150 steps; movement 3.3 / 8.1 / 12.4 / 22.2%. Used **lr=2e-2** (fastest
  mover, also lowest F). Movement is slower than the full-batch dissociation bench because training is
  **batch-1** (one pair per step → directions partly cancel), so ||W−W₀|| grows sublinearly.

## Metric discipline

F (energy) is **not** a success signal — in PC, relaxation drops F while weights coast. The decisive
demonstration is in these very runs: **the relu run reached a *lower* final F (0.038) than the leaky run
(0.089), yet the relu run's text→image was total mush and the leaky run's generates.** F-descent is here
*anti-correlated* with generation. Judgment is on weight movement, generation diversity, and per-sample
recognizability — never F.

---

## Results

| metric | **Run 1 — ReLU** (lr 2e-2) | **Run 2 — leaky-ReLU** (lr 2e-2) |
|:--|--:|--:|
| weight movement (overall) | 210% | **603%** |
| F: start → end | 0.737 → **0.038** | 0.737 → 0.089 |
| **text→image diversity ratio** | **0.000** (collapse) | **0.430** (varies) |
| **text→image retrieval top-1** (chance 0.016) | 0.016 (chance) | **0.297** (≈19× chance) |
| text→image beats-mean | 0.000 | 0.203 |
| text→image output range across captions | 0.000 (identical) | 0.567 (varies) |
| image→image recon MSE | 0.0045 | 0.0299 |
| image→text token acc (baseline 0.045) | 0.980 | 0.352 (≈8× baseline) |
| text-encoder taps `tt` std across captions | **[0,0,0,0] (DEAD)** | [0.29, 4.14, 0.28, 0.27] (alive) |

Per-layer movement (leaky run, weight tensors): image enc c1=61%, c5=92%, Wi1=72%; text enc emb=90%,
Wt1=45%, Wq0=115%; decoders proj/W_DI moved most (tiny 1e-3 init). **Every layer moved** — both encoders
alive — i.e. a genuinely trained model, unlike the 7.7B's ~1.4%.

### Run 1 (plain ReLU): a dead text encoder, not undertraining

Run 1 trained *hard* (210% movement, F→0.038) and the image pathway works beautifully (image→image MSE
0.0045; image→text 0.98). But **text→image diversity was exactly 0.000** — every caption decoded to the
*same* image. The diagnostic localized it precisely: the **text-encoder tap heads were dead** —
`tt std=[0,0,0,0]`, `mean-norm=[0,0,0,0]` for all captions — a classic **dying-ReLU**: the aggressive lr
drove their pre-activations negative, ReLU zeroed them, and dead units get no gradient to recover. The
text *decoder* still worked (image→text 0.98) because it read the image-informed shared latent; only the
text *encoder* died, so text→image had no per-caption signal. This is a dead-unit artifact — **not**
undertraining, and **not** the decoder failing (image→image proves the decoder produces varied, correct
images). It is *not a valid positive control*, because half the model never trained.

### Run 2 (leaky-ReLU, the only change): generates in all directions

Leaky-ReLU (α=0.01) prevents unit death; everything else identical. The text encoder came alive (taps
vary by caption), weights moved 603%, and **text→image now varies and is recognizable**: diversity 0.43
(well above the 0.30 collapse threshold), retrieval 0.30 (≈19× chance), outputs differ per caption
(range 0.57). image→image still reconstructs (0.030) and image→text stays 8× above baseline (0.35; lower
than run 1's 0.98 because, with a live text encoder, the shared latent serves both modalities rather than
overfitting the image→text path). See `midscale_grid.png`.

---

## Verdict and honest scope

**The full bidirectional design GENERATES when properly trained.** When the weights actually move (603%)
and both encoders are alive, text→image produces varying, recognizable-above-chance output — not the gray
mush the undertrained 7.7B gave. This is the positive-control prerequisite for the paper: the
architecture is capable of conditional generation; the 7.7B failure is consistent with undertraining.

**Caveats — do not overclaim:**

- **Recognizable, not crisp.** Retrieval 0.30 (19× chance) and diversity 0.43 are clearly non-collapsed
  but far from the retrieval=1.00 of the tiny single-scale dissociation bench. This is a 64-pair,
  batch-1, 50M, ~10-min CPU run; "recognizable-but-imperfect" is the honest description. Longer training
  / full-batch / richer data would be needed for crisp samples.
- **Two distinct "mush" failure modes, both training-side, were observed:** (a) **undertraining** —
  weights frozen near init → everything mush (the 7.7B case); (b) **dying-ReLU in the text encoder** —
  text taps die under aggressive lr → text→image collapses while the image pathway still works (run 1).
  "Move the weights" addresses (a); avoiding unit death (leaky-ReLU, or a gentler lr) addresses (b). The
  paper should treat the text→image pathway as fragile, not automatically cured by weight movement alone.
- **Leaky-ReLU is a deviation from the relu design**, made to fix the diagnosed dead-unit pathology, not
  to chase a positive — and reported as such, alongside the relu run. It does not change the
  bidirectional / energy / LARS / Option-C structure.
- **Scale.** This is mid-scale (50.5M), not the 7.7B. It shows the design *can* generate at a size whose
  weights move; it does not prove the 7.7B would generate if trained longer (that remains the at-scale
  positive control the literature review flags). But it removes "the architecture fundamentally cannot
  generate" as the explanation for the 7.7B mush.

---

## Addendum — resolving the dying-ReLU asterisk: activation × LR grid

To determine whether the dying-ReLU is an LR artifact or an architectural fragility, and to test the
hypothesis that **GELU** (smooth, no hard zero — and the activation implied by the original model's
`d_gelu`-derivative-on-a-ReLU-forward bug) avoids it, I ran the full grid **{ReLU, leaky-ReLU, GELU} ×
{aggressive lr 2e-2, gentle 6.7e-3 (1/3), gentle 2e-3 (1/10)}**, everything else identical to the 50.5M
model, fresh identical init per cell, 2500 steps each (`midscale_actgrid.py`, ~63 min CPU). "Tap std" =
std of the text-encoder taps across captions (≈0 ⇒ dead; the encoder emits the same vector for every
caption). Chance retrieval = 0.016.

| activation | lr | weight move | text-enc tap std | state | text→img diversity | text→img retrieval | recon | img→text |
|:--|:--|--:|--:|:--|--:|--:|--:|--:|
| ReLU | 2e-2 (aggr) | 149% | **0.000** | **DEAD** | 0.000 | 0.016 (chance) | 0.009 | 0.904 |
| ReLU | 6.7e-3 (1/3) | 17.7% | 0.007 | ~dead | 0.055 | 0.016 | 0.068 | 0.080 |
| ReLU | 2e-3 (1/10) | 8.0% | 0.013 | alive | 0.060 | 0.016 | 0.069 | 0.080 |
| leaky-ReLU | 2e-2 | 196% | 0.020 | alive | 0.441 | 0.000 | 0.066 | 0.078 |
| leaky-ReLU | 6.7e-3 | 22.1% | 0.012 | alive | 0.034 | 0.016 | 0.068 | 0.082 |
| leaky-ReLU | 2e-3 | 10.3% | 0.020 | alive | 0.046 | 0.016 | 0.069 | 0.082 |
| **GELU** | **2e-2** | 106.6% | 0.014 | **alive** | **0.397** | **0.703** (≈45× chance) | **0.020** | **0.559** |
| GELU | 6.7e-3 | 39.3% | 0.017 | alive | 0.040 | 0.016 | 0.065 | 0.072 |
| GELU | 2e-3 | 12.8% | 0.029 | alive | 0.035 | 0.016 | 0.069 | 0.086 |

Figures: `actgrid_heatmap.png` (diversity + movement over the grid), `actgrid_samples.png` (text→image
strips; dead cells show identical mush). Raw: `actgrid_results.json`.

### What the grid shows (blunt)

1. **The dying-unit collapse is a HARD-ZERO-activation problem.** ReLU is the *only* activation that dies;
   its tap std goes 0.000 → 0.007 → 0.013 as lr drops from 2e-2 → 6.7e-3 → 2e-3 (more death at higher
   lr). **leaky-ReLU and GELU never die at any lr** (tap std 0.012–0.029 throughout). Removing the hard
   zero removes the death. This is the clean general statement.

2. **The death is LR-coupled, but plain ReLU has NO good operating point.** ReLU survives only at the
   gentlest lr (2e-3) — where the weights move just 8% (undertrained → diversity 0.06, no generation).
   The lr that moves weights enough to generate (aggressive 2e-2, ≥100% movement) is exactly where ReLU
   dies. So "dying-ReLU is just an LR artifact" is *too generous*: there is no plain-ReLU lr in this grid
   that simultaneously (a) moves the weights enough and (b) keeps the encoder alive. ReLU is caught
   between dying (high lr) and undertraining (low lr).

3. **Generation needs BOTH high movement AND a live (smooth) activation.** Only the aggressive-lr cells
   move weights >100%; of those, **GELU generates recognizably** (retrieval 0.70 ≈ 45× chance, diversity
   0.40, best image side too: recon 0.020, img→text 0.56). leaky-ReLU at aggressive lr *varies* (diversity
   0.44) but is **not yet recognizable at 2500 steps** (retrieval 0.00; cf. the 4000-step leaky run above
   which reached retrieval 0.30 — leaky needs more steps; GELU is more sample-efficient here). All
   gentle-lr cells are undertrained (8–39% movement) regardless of activation — the movement/undertraining
   axis, consistent with the rest of this work.

4. **GELU is the standout and is consistent with it being the intended activation** (the original
   `d_gelu`-on-ReLU bug). With GELU the architecture both trains (weights move) and generates, with no
   dying-unit fragility.

### Verdict

**The dying-unit collapse is a hard-zero-activation pathology (ReLU only) that bites precisely in the
high-lr / high-movement regime required to train; it is not a limit of the architecture.** Smooth
activations remove it: **GELU (and leaky-ReLU) keep the text encoder alive at every lr, and GELU at the
training-strength lr generates recognizably (retrieval 45× chance).** For the model this means: use a
smooth activation (GELU preferred) — then "move the weights" is the whole story, and the two mush modes
collapse into one (undertraining), with no separate dying-unit caveat.

Honest limits: single seed per cell; 2500 steps (leaky's recognizability is still climbing at this
budget — see its 4000-step datapoint above); 50.5M / MNIST / batch-1 / CPU. GELU's retrieval 0.70 is the
strongest generation in any run here but should be confirmed with more seeds/steps before headline use.

---

## Addendum 2 — confirmatory multi-seed replication of the headline GELU result

The retrieval-0.70 GELU number was one seed at 2500 steps. Before it goes in the paper, replicated with
**GELU and leaky-ReLU at lr=2e-2, full 4000 steps, 3 independent seeds each** (seed drives weight init,
the MNIST subset, the captions, and the train order — fully independent replicates), `midscale_seeds.py`,
~70 min CPU. Chance retrieval = 0.016.

| activation | seed | weight move | text→img diversity | **text→img retrieval** | recon | img→text |
|:--|:--|--:|--:|--:|--:|--:|
| GELU | 0 | 226% | 0.464 | 0.891 | 0.017 | 0.672 |
| GELU | 1 | 146% | 0.542 | 0.844 | 0.012 | 0.805 |
| GELU | 2 | 163% | 0.521 | 0.938 | 0.012 | 0.818 |
| **GELU** | **mean ± std** | 178 ± 35% | **0.509 ± 0.033** | **0.891 ± 0.038** | 0.014 | 0.765 |
| leaky | 0 | 199% | 0.174 | 0.031 (≈chance) | 0.052 | 0.141 |
| leaky | 1 | 263% | 0.370 | 0.531 | 0.032 | 0.365 |
| leaky | 2 | 365% | 0.478 | 0.609 | 0.022 | 0.498 |
| leaky | **mean ± std** | 275 ± 68% | 0.341 ± 0.126 | **0.391 ± 0.256** | 0.035 | 0.335 |

Best-GELU-seed samples: `seeds_gelu_grid.png`. Raw: `seeds_results.json`.

### Verdict — REPLICATES

**GELU's recognizable generation replicates robustly: retrieval 0.89 ± 0.04 across 3 independent seeds
(every seed ≥ 0.84 = 54× chance), diversity 0.51 ± 0.03.** This is the number for the paper. It is *not*
a one-seed fluke — and it is *stronger* than the single 2500-step seed (0.70), because at the full 4000
steps the model is better trained. Variance across seeds is low.

**GELU is genuinely more sample-efficient and more reliable than leaky-ReLU at matched budget.** At 4000
steps leaky reaches only 0.39 ± 0.26 and is **bimodal/unreliable** — one seed failed at chance (0.031),
the others reached ~0.6. So leaky does *not* cleanly catch up given the steps; GELU wins on both mean and
variance. (Caveat on the per-cell "DEAD" tag in the script: the `tap_std < 1e-2` threshold is calibrated
to ReLU's hard-zero death and mislabels leaky's low-but-nonzero tap variance; leaky units do not hard-die
— leaky's failure mode here is generation *variance*, not unit death.)

**Net for the paper:** GELU is the activation. With it, the mid-scale bidirectional design generates
recognizably and reproducibly (retrieval 0.89 ± 0.04, ~54× chance), with no dying-unit fragility — so the
clean single-axis story holds: the 7.7B mush was undertraining (weights must move), and at an appropriate
LR with a smooth activation the architecture generates. Honest limits unchanged: 50.5M / MNIST / batch-1 /
CPU, 64-pair memorization; this establishes capability and reproducibility, not at-7.7B-scale generation.

---

## Addendum 3 — staged pretraining vs from-scratch, on REAL image–text

**Question (a different lever from the activation work above):** does pretraining the image and text
branches *separately as autoencoders* first, then assembling + sharing latents + joint-training,
generate **better or faster** than training the same model jointly from scratch? This is the cheap
mid-scale test of staged pretraining before betting it on the 7.7B at-scale positive control.

**Verdict (blunt): NO — staged pretraining does NOT help, and the autoencoder features *fight* the
cross-modal coupling at assembly. A null/negative result, reported as one (2 seeds).** At matched
seed/init/params/total-budget, the staged arm is **equal-or-worse than from-scratch on every metric**,
and the mechanism is unambiguous: independently trained AEs learn good *unimodal* features that are
**not cross-modally aligned** (tap-alignment at assembly = chance), so when the branches are first
forced to share a latent the joint energy *explodes* (Arm B's first joint-phase F ≈ **3.4e3 (seed 0) /
1.2e7 (seed 1)** vs Arm A's ~0.17) and the short joint phase spends its budget reconciling that
mismatch instead of building the coupling — ending *more collapsed* than from-scratch.

**Data — real images + real WORD text (not MNIST / digit tokens):** CIFAR-100, **one image per fine
class (N=100 distinct 32×32×3 natural images)**, each captioned with its real class name,
`"a photo of a {class}"` (char-level one-hot, T=26, V=27). COCO was the stated preference but its path
(241 MB annotation zip + per-image scraping) is too heavy/risky for a hard CPU time-box and would
starve the two training arms; CIFAR-100-one-per-class still gives genuinely **distinct** real images
each paired with a **distinct real-word** caption (CIFAR-10's repeated class captions would be unusable
for retrieval). Chance retrieval = 0.010. `midscale_staged.py`, CPU, ~15 min/seed.

**Two arms, matched on everything except staging** (same seed → identical init, same params, same total
step budget; only the procedure differs):
- **Arm A — from scratch:** assemble the full bidirectional model, joint-train from random init, 3000 steps.
- **Arm B — staged:** Phase 1 image-AE (1000 steps, image enc+dec only), Phase 2 text-AE (1000 steps,
  text enc+dec only), Phase 3 assemble + joint-train (1000 steps). Total = 3000, matching Arm A.

Recipe identical to the validated model above (GELU, lr=2e-2, one energy F, LARS + bias floor,
relax(8)-then-step, dense per-scale anchors, A_GEN=2.0 ≥ A_CROSS=1.0). 62.5M params (slightly larger
than the MNIST model: 3-channel 32×32 input).

### Results (per seed; raw in `midscale_staged_results_seed{0,1}.json`)

| metric | **A (scratch)** s0 / s1 | **B (staged)** s0 / s1 |
|:--|--:|--:|
| text→image retrieval (chance 0.010) | 0.010 / 0.020 | 0.010 / 0.010 |
| **text→image diversity** (varies by caption?) | **0.335 / 0.301** | 0.071 / 0.125 (more collapsed) |
| image→image recon MSE | **0.0315 / 0.0331** | 0.0487 / 0.0468 (worse) |
| image→text token acc | 0.792 / 0.793 | 0.779 / 0.759 |
| weight movement overall | 82% / 77% | 86% / 81% |
| **Phase-3 joint F at start** | (0.17 / 0.18) | **3.4e3 / 1.2e7** |
| cross-modal tap-alignment @ assembly (chance 0.010) | — | **0.010 / 0.010** |

Grids: `midscale_staged_grid.png` (seed 0; rows = target / A txt→img / B txt→img / A img→img / B
img→img), `midscale_staged_grid_seed1.png`.

### The key diagnostic — do the AE features help or fight?

1. **They do NOT pre-align the modalities.** Cross-modal tap-alignment (does each caption's text-tap
   retrieve its own image-tap?) at assembly = **0.010 = chance** for both seeds — identical to random
   init — even though the image-AE reconstructs (recon 0.043–0.065) and the text-AE works (token-acc
   0.77–0.78). The AEs learn good *within-modality* features with *zero* cross-modal correlation. This
   is exactly the pre-registered risk: AE features are tuned for reconstruction, not for the coupling.

2. **They fight at assembly.** Because the branches are independently specialized, forcing them to
   share a latent produces an enormous energy mismatch — **Arm B's first joint-phase F ≈ 3.4e3 (seed 0)
   to 1.2e7 (seed 1)** vs Arm A's ~0.17. The 1000-step joint phase then burns its budget crushing that
   mismatch rather than learning the coupling, and ends **more collapsed** (diversity 0.07–0.13) than
   from-scratch (0.30–0.34).

3. **The pretraining is net wasted.** Arm A (no image-AE) reaches **better** final image recon
   (0.032) than staged Arm B (0.047), so the dedicated image-AE phase bought no final-recon advantage;
   the 2000 AE steps would have been better spent on joint training, as Arm A demonstrates. (For
   accuracy: the image-AE recon is *internally* preserved through Arm B's joint phase — 0.066→0.047 on
   seed 1 — so the "fight" is in the **cross-modal coupling**, not in the joint phase destroying
   unimodal recon.)

### Honest caveats

- **Both arms were below the recognizability bar:** text→image retrieval was at/near chance
  (0.010–0.020) for *both* — real CIFAR-100 RGB at 3000 CPU steps is far harder than the MNIST regime
  where this recipe reached retrieval 0.89. So this is "staged vs scratch, both blobby," and the
  comparison rests on the **secondary** signals (diversity, recon, alignment, assembly energy), which
  uniformly favor from-scratch. The diversity gap (A ≈0.32 vs B ≈0.10) and the 10³–10⁷× assembly-energy
  mismatch are the load-bearing evidence, not the floored retrieval.
- 2 seeds, 3000 steps, 62.5M, CPU, N=100 memorization. Establishes direction, not a tuned ceiling.
- This tests **staged AE-pretraining as specified** (independent unimodal AEs → assemble). It does
  *not* rule out staged schemes that pretrain the *coupling itself* (e.g. a cross-modal
  contrastive/InfoNCE warm-up), which would by construction align the very taps the unimodal AEs leave
  at chance.

### Net for the 7.7B

**Staged AE-pretraining alone is not the at-scale fix.** It gives no cross-modal head start (the AEs
leave the modalities unaligned) and creates an assembly-time energy mismatch the joint phase must undo.
The earlier finding stands as the live lever: the 7.7B mush was **undertraining** (the weights must
*move*, with a smooth activation). If a warm-start is still wanted, it must pretrain the cross-modal
**coupling**, not the two modalities in isolation.
