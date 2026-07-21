# Step 1 gate, held-out re-test: does the caption to image map generalize, or was the gate memorization?

The Step 1 gate (00304a9) computed retrieval, diversity, recon on the SAME 400 pairs it trained on, so
it could not separate memorization from generalization. This run (`run_step1_coco_heldout.py`, bdeee90)
is byte-identical in recipe (GELU, plain LARS, lr=2e-2, single energy F, N_INFER/GEN_INFER, widths,
init, 156.6M, 5000 steps) and changes only the data split and the readout scope: train on 400 pairs,
evaluate text to image retrieval / image to image recon / diversity / i2t on a DISJOINT 100 pairs the
model never saw. Vocab built from train captions only (no eval leakage). A100, ~24 min train, weights
moved 116 percent (well past the 40 percent validity floor, so the verdict is valid, not undertrained).

## Verdict (pre-registered rule, no goalpost moving): HELD-OUT PARTIAL

On unseen captions the output VARIES (eval diversity 0.27 >= 0.20, out-range 0.34 > 1e-2) but
text to image retrieval is NOT above chance: **2 hits out of 100, chance expects 1** (the pre-registered
PASS bar was > 3 hits, roughly 2 sigma). So by the rule fixed before the run, this is PARTIAL: the
caption to image map fit the training pairs but did not generalize to a statistically above-chance
retrieval result on unseen captions. **The gate's headline retrieval claim does not survive out of
sample.**

## Train vs held-out (the gap is the result)

| metric | train (N=400, chance 0.0025) | held-out (N=100, chance 0.0100) | gap (train - eval) |
|:--|--:|--:|--:|
| text to image retrieval | 0.020 = **8/400** (~7 sigma, significant) | 0.020 = **2/100** (~1 sigma, NOT significant) | +0.000 |
| diversity ratio | 0.375 | 0.273 | +0.102 |
| out-range | 0.603 | 0.338 | +0.265 |
| image to image recon MSE | 0.0293 (train-mean baseline 0.0654) | **0.0312 (baseline 0.0605, beats it)** | -0.0019 |
| image to text token acc | 0.351 (baseline 0.200) | 0.308 (baseline 0.218) | +0.043 |

The raw retrieval RATE is identical train and eval (0.020), and the recon and i2t gaps are tiny. That
matters: this is NOT a classic train-overfit blowup, where train retrieval would dwarf eval. The
mechanism behind the gate's inflated headline is the retrieval-POOL size, not per-pair memorization. A
weak ~2 percent absolute hit rate, scored against a 400-way in-sample pool (chance 0.0025), reads as
"8x to 10x chance, ~7 sigma" and looks like a real signal; the SAME 2 percent rate scored against the
100-way held-out pool (chance 0.01) is just 2 hits, within noise of the 1 expected. So the gate's
"10x chance" was real in-sample but is a pool-size artifact for the purpose of claiming generalization.

## What the held-out grid shows (honest)

`step1_heldout_grid.png` rows are target / text-to-image / image-to-image, all on the unseen 100. The
text-to-image row is brown and tan blobby color fields that vary only weakly across captions and are
visibly more uniform than the in-sample gate grid (consistent with eval diversity 0.27 vs train 0.375).
They are not recognizable scenes. The image-to-image row reconstructs the unseen images coarsely but
beats the train-mean baseline.

## What this establishes, and what it does not

Establishes (real, generalizes out of sample):
- Image to image reconstruction generalizes: eval recon 0.031 beats the train-mean baseline 0.061 and
  about equals train recon 0.029. The autoencoding side works on unseen images.
- Text to image output on unseen captions VARIES (diversity 0.27, out-range 0.34, not collapsed), and
  the train-minus-eval gap is small on every metric.

Does NOT establish (the load-bearing negative):
- Above-chance caption to image RETRIEVAL on unseen captions. 2 of 100 is not above the 1 expected by
  chance and fails the pre-registered > 3 bar. The model does not demonstrate that it matches an unseen
  caption to its correct image better than chance.

Bottom line, not softened: the gate's "real text to image generation, 10x chance" should be downgraded.
Out of sample, the model reconstructs and produces caption-varying blobs, but it does not match unseen
captions to images above chance. The in-sample 10x was inflated by the 400-way retrieval pool. The
conference claim that survives is the weaker one: weights moving breaks mode-collapse and the image side
generalizes, but above-chance text to image matching is in-sample only and not demonstrated on unseen
captions at this scale and budget.

Artifacts: `step1_heldout_grid.png`, `step1_heldout_results.json`, `run_step1_coco_heldout.py` (bdeee90).
