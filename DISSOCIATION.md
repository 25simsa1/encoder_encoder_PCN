# Dissociation Matrix — generative mode-collapse in a bidirectional PCN is UNDERTRAINING

**Claim under test:** In a bidirectional image–text predictive coding network, mode-collapse to the
dataset mean is caused by **insufficient weight movement (undertraining)**, *not* by latent capacity or
the training objective. **Prediction:** only the weight-movement axis flips collapse — widening the
latent or adding an anti-mean loss does **not** fix it under low movement, and high movement fixes it
**regardless** of latent width or anti-mean loss.

**Verdict (blunt): THE CLAIM HOLDS.** Across the full 2×2×2 factorial, all 4 low-movement cells collapse
(diversity ≤ 0.07, retrieval = chance) regardless of latent width or anti-mean loss, and all 4
high-movement cells break collapse (diversity ≥ 0.52, retrieval = 1.00) regardless of latent width or
anti-mean loss. Only weight movement moves the outcome. See `dissoc_matrix.png`, `dissoc_curve.png`,
`dissoc_samples.png`, `dissoc_Ftraj.png`, raw metrics in `dissoc_results.json`. Reproduce:
`python3 dissociation.py` (CPU, ~15 min).

---

## Metric discipline (read first)

**F (energy) decreasing is NOT evidence of learning** and is never used as a success signal. In PC, F
drops via *state relaxation* absorbing per-sample error while the weights coast — that is precisely the
phenomenon being documented. F is reported only to show it drops in *every* cell, so F-descent cannot
distinguish collapse from success.

Load-bearing metrics (decide every cell), defined precisely:

- **Weight movement** = ‖W − W₀‖ / ‖W₀‖ as a percentage, computed **overall** (global ratio over all
  parameters) and **per-layer** (per weight tensor; zero-initialized biases are excluded because the
  ratio is undefined for ‖W₀‖≈0 — their movement is still counted in the overall ratio).
- **Diversity ratio** = mean over pixels of `std(text→image generations across the N captions)` divided
  by the dataset's own pixel-std. Collapse ⇒ ~0 (every caption decodes to the same image). Varies-by-input
  ⇒ a real fraction of 1. **Collapse threshold τ = 0.30**, applied uniformly. (The margin is large —
  collapsed cells ≤ 0.07, broken cells ≥ 0.32 — so the verdict is insensitive to τ.)
- **Recognizability (retrieval top-1)** = for each text→image generation, find the nearest target image by
  MSE; accuracy = fraction whose nearest target is its *own* target. **Chance = 1/N = 0.031.**
- **Recognizability (beats-mean)** = fraction of samples whose generation is closer (MSE) to its own
  target than to the dataset mean image.
- **Recon MSE** = image→image reconstruction error (sanity check).

---

## Bench

- **Data:** N = 32 distinct MNIST images (resized 20×20), each paired with a **distinct random caption**
  (6 tokens, vocab 16). No shared class-label shortcut — text→image must carry per-sample content. Dataset
  pixel-std = 0.297.
- **Model (bidirectional, validated lineage `port_redesign.py` → `port_redesign2.py`):** conv image encoder
  (16→32 ch) + 1-block transformer text encoder (d_model 64, 4 heads, FFN 128) → shared latent S; image
  decoder (hidden 512) + text decoder. Per-edge **mean-MSE** energy. **LARS (trust-ratio) relax-then-step**:
  inner loop relaxes S by gradient descent on the energy (12 steps), then one LARS weight update.
- **Scale:** narrow latent (16) ⇒ **268,560** params; wide latent (512) ⇒ **999,664** params. Not a toy;
  CPU-affordable.
- **Controls:** fresh model per cell, **identical weight init (seed 42)** and identical data; everything
  held fixed except the single swept axis. 1000 training steps per cell.
- **Axes:** A1 latent width {narrow 16, wide 512}; A2 anti-mean {off, InfoNCE contrastive on the two
  latents, strength 1.0}; A3 weight movement {LOW: LARS lr 1e-5 → weights coast; HIGH: LARS lr 8e-3 →
  weights move tens–hundreds of %}. A3 is operationalized as the **effective learning rate**, consistent
  with the claim that too-small effective LR (relative to the step budget) is the root cause.

---

## The dissociation matrix (8 cells)

| weight-move | latent | anti-mean | **move %** | **diversity** | **retrieval** (chance .03) | beats-mean | recon MSE | F: start→end | **outcome** |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| **LOW** | narrow(16) | off | 1.0 | 0.039 | 0.03 | 0.00 | 0.219 | 0.330→0.290 | **COLLAPSE** |
| **LOW** | narrow(16) | on  | 1.0 | 0.052 | 0.03 | 0.00 | 0.219 | 0.330→0.304 | **COLLAPSE** |
| **LOW** | wide(512)  | off | 1.0 | 0.065 | 0.03 | 0.00 | 0.210 | 0.261→0.237 | **COLLAPSE** |
| **LOW** | wide(512)  | on  | 1.0 | 0.069 | 0.03 | 0.00 | 0.212 | 0.261→0.241 | **COLLAPSE** |
| **HIGH** | narrow(16) | off | 172.6 | 0.523 | 1.00 | 0.97 | 0.0028 | 0.325→0.024 | **VARIES** |
| **HIGH** | narrow(16) | on  | 315.2 | 0.553 | 1.00 | 0.94 | 0.0010 | 0.325→0.022 | **VARIES** |
| **HIGH** | wide(512)  | off | 52.0  | 0.561 | 1.00 | 1.00 | 0.0003 | 0.251→0.0018 | **VARIES** |
| **HIGH** | wide(512)  | on  | 106.8 | 0.573 | 1.00 | 1.00 | 0.0001 | 0.251→0.0002 | **VARIES** |

Reading the matrix (`dissoc_matrix.png` shows it as two heatmaps):

- **A1 (latent width) does NOT flip collapse.** Under LOW movement, going narrow(16)→wide(512) leaves
  diversity at 0.04→0.07 and retrieval at chance — still collapsed. A 512-dim latent does not rescue an
  undertrained model.
- **A2 (anti-mean loss) does NOT flip collapse.** Under LOW movement, adding the InfoNCE contrastive
  objective leaves diversity 0.04→0.05 and retrieval at chance — still collapsed. The anti-mean loss
  cannot act because at lr 1e-5 it cannot move the weights.
- **A3 (weight movement) flips collapse, every time.** Every HIGH cell breaks collapse (diversity ≥ 0.52,
  retrieval = 1.00, beats-mean ≥ 0.94) — for narrow *and* wide latents, with anti-mean *off and on*.

## Weight-movement-vs-collapse curve (fixed arch: narrow latent, anti off; sweep effective LR)

`dissoc_curve.png` — diversity ratio as a function of weight movement, sweeping LARS lr over 8 values:

| LR | move % | diversity | retrieval | outcome |
|:--|--:|--:|--:|:--|
| 1e-5 | 1.0  | 0.039 | 0.03 | collapse |
| 3e-5 | 2.7  | 0.018 | 0.03 | collapse |
| 1e-4 | 7.8  | 0.047 | 0.03 | collapse |
| 3e-4 | 21.7 | 0.062 | 0.03 | collapse |
| 1e-3 | 47.6 | 0.323 | 0.53 | **breaks** |
| 3e-3 | 105.2 | 0.458 | 1.00 | breaks |
| 8e-3 | 172.6 | 0.523 | 1.00 | breaks |
| 2e-2 | 515.0 | 0.552 | 1.00 | breaks |

Monotone relationship with a clear **threshold: collapse breaks between ~22% movement (still collapsed,
retrieval = chance) and ~48% movement** (diversity crosses τ, retrieval jumps to 0.53). Above ~100%
movement, retrieval saturates at 1.00 and diversity at ~0.55. Below ~20% movement the model is
indistinguishable from the dataset mean regardless of how much F has dropped.

## Per-layer movement (weight tensors)

| cell | overall | image dec di0 / di1 | text dec dt | img enc wi | txt enc wt | conv c1 / c2 |
|:--|--:|--:|--:|--:|--:|--:|
| LOW/narrow/off  | 1.0%   | 1% / 1%     | 1%   | 1%  | 1%  | 1% / 1%   |
| HIGH/narrow/off | 172.6% | 261% / 137% | 276% | 19% | 16% | 12% / 19% |
| HIGH/wide/off   | 52.0%  | 86% / 63%   | 38%  | 31% | 35% | 15% / 23% |

LOW freezes **every** layer uniformly at ~1% — the model never leaves its initialization. Under HIGH
movement the **decoders move the most** (di0, di1, dt), which is the mechanistic locus: a near-init
decoder maps any latent to a near-constant output (the dataset mean); moving the decoder is what lets
distinct latents decode to distinct images.

## F is not the signal (`dissoc_Ftraj.png`)

F decreases in **every** cell, including collapsed ones. The sharpest demonstration is in the curve: at
7.8% movement F falls 0.330→0.216 and at 21.7% movement F falls 0.330→0.108 (a ~3× drop) — yet both are
fully collapsed (retrieval = chance, diversity < 0.07). Meanwhile the LOW matrix cells drop F via
relaxation (0.330→0.290) with frozen weights. **A large F-descent is fully consistent with total
collapse.** The overlaid trajectories (LOW-narrow vs HIGH-narrow) both fall; only the high-movement run
acquires per-input variation.

---

## Verdict

**The dissociation holds cleanly.** Mode-collapse-to-the-mean in this bidirectional image–text PCN is an
**undertraining / weight-movement** phenomenon. Latent capacity (8×) and an explicit anti-mean objective
**both fail** to break collapse when weights barely move; raising the effective learning rate so weights
actually move **always** breaks it, narrow or wide latent, anti-mean off or on. F-descent tracks neither.

This is the dissociation the literature review (`LIT_REVIEW.md`) identified as the novel contribution no
prior PC paper has run: prior work attributes such collapse to architecture / energy-landscape
(Bidirectional PC, arXiv:2505.23415) or inverse-problem underdetermination (Millidge review,
arXiv:2107.12979), and the weight-movement *mechanism* alone (arXiv:2305.13562) was never connected to
generative mean-collapse or tested against the latent-capacity and objective alternatives.

### Honest scope / caveats

- **Scale.** This is a controlled small-scale dissociation (MNIST, N = 32, 0.27–1.0M params, CPU). It
  establishes the *causal structure* (only movement flips collapse). It does **not by itself** prove the
  motivating 7.7B run was undertrained — that requires the positive control *at scale* (raise effective
  movement on the large model and show collapse resolves), as flagged in `LIT_REVIEW.md`.
- **A3 = effective LR.** Weight movement is manipulated via the LARS learning rate (1e-5 vs 8e-3). This is
  deliberate: the claim is that too-small *effective* LR is the cause, so LR is the correct knob. The LOW
  setting (~1% movement) reproduces the large-model "weights coast near init" regime.
- **What would have refuted the claim:** any LOW cell with a wide latent or anti-mean loss breaking
  collapse, or any HIGH cell failing to. Neither occurred. Had they, this doc would report the
  dissociation as failed — the verdict is not tuned to the desired outcome (the only quantity tuned was
  the HIGH learning rate, chosen so weights *move*, which is the definition of the axis, not the outcome).
