# Step 1 gate — real-COCO text→image at scale (the make-or-break run)

The conference gate. Re-attacks exactly what mushed in 7c (real COCO captions + real images), but with the
Run-A/B fixes: GELU + **plain LARS** (no muP) + the weight-moving **lr=2e-2**, scale-matched (157M, real
step budget on GPU), unlike the earlier real-data attempts (CIFAR-100 staged, COCO 7c) which were
undertrained/small (<=62M, 3k steps, CPU).

Ran `run_step1_coco_gate.py` (committed ddde423) on a RunPod **A100 80GB**. Data: COCO val2017 subset,
**N=400** image-caption pairs, **real lowercased sentence captions** (char-level one-hot, CAPLEN=64,
V=32), images **64x64x3** in [0,1] RGB. Model **156.6M params** (DM=768, DIMS=[3072,3072,1536,1536]).
Recipe: single energy F, GELU, plain LARS + bias floor, relax-then-step, dense multi-scale anchors,
A_GEN>=A_cross, all grads via GradientTape. lr=2e-2, **5000 steps**, local-disk checkpoints. Chance
retrieval = 1/400 = 0.0025.

## Result — GATE PASS

| metric | value | read |
|:--|--:|:--|
| weight-movement | **107.6%** | far past the 40% gate; not undertrained, verdict valid |
| text→image diversity | **0.360** | varies by caption (0 = collapse) |
| text→image retrieval top-1 | **0.025** (chance 0.0025) | **10x chance** (above chance) |
| text→image out-range | **0.541** | outputs DIFFER by caption (0 = identical-for-all = the 7c failure) |
| image→image recon MSE | 0.031 | reconstructs |
| image→text token acc | 0.350 (baseline 0.200) | above baseline |
| training | 5000 steps, 26.5 min, no divergence, F 0.115→0.039 | stable |

**Verdict: real-text→image generates at scale.** With the weights actually moving (108%), text→image
produces output that **varies by caption** (diversity 0.36, out-range 0.54) and is **above chance**
(retrieval 0.025 = 10x chance). This is the decisive contrast with 7c, where every caption produced the
**identical** gray/brown image (out-range ~0, mode-collapse). The collapse is broken.

## Honest calibration (what "pass" means here)

The generated images are **blobby color/texture fields, not crisp recognizable scenes** — at 64x64,
157M, 5000 steps this is expected and was the pre-registered bar ("judge on varies-by-caption + above
chance, NOT prettiness"). In the grid (`step1_coco_grid.png`, rows: target / text→image / image→image)
the text→image row shows fields that clearly differ per caption in dominant color, brightness, and
light/dark layout, but you cannot yet read the described object off them. So the honest statement is
**"the image responds to the text, above chance"**, not "it synthesizes the described scene." Retrieval
10x chance is real but modest.

What this establishes and does not:
- **Establishes:** the validated recipe, scale-matched with weights moving, does NOT mode-collapse on
  real text + real images the way the undertrained 7c did. Real-text→image conditional generation is
  happening (caption-dependent, above chance). The 7c mush was undertraining, confirmed on the real task.
- **Does not:** produce crisp/recognizable images. Retrieval is 10x chance, not near 1.0. Getting from
  "responds to text" to "renders the scene" is the next problem (more steps, higher resolution, larger
  model, or a better decode pathway), and is where a warm-up or architectural change might still matter.

Artifacts: `step1_coco_grid.png`, `step1_coco_results.json`, `run_step1_coco_gate.py` (ddde423).
