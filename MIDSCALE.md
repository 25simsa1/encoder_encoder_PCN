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
