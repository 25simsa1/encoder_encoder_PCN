# Alignment Without Binding: Backpropagation Acquires Transferable Cross-Modal Structure Where Local Predictive-Coding Learning Does Not

Draft v2 (2026-07-23). Supersedes PAPER_DRAFT.md's framing; every number traces to a committed
artifact (Appendix A). Target: ICLR 2027 main (TMLR fallback).

## Abstract

Can local, backprop-free learning rules acquire cross-modal structure that generalizes? We
train byte-matched image-text architectures (156M parameters, COCO, 20k pairs, from scratch)
under two learning rules -- end-to-end backpropagation on a symmetric InfoNCE objective, and a
predictive-coding (PC) style local rule that minimizes the same contrastive coupling through
per-layer prediction-error updates -- and compare them at MATCHED TRAINING FIT: both rules
memorize the training pairing near-perfectly (latent retrieval 0.95-1.0). Held out, they
dissociate. Backprop acquires transferable CATEGORY-LEVEL cross-modal structure: held-out
image-to-caption category precision@10 of 20.0-21.5% against an 8.1% base rate (2.43-2.64x
lift, three seeds, systematic across categories: planes 6.1x, cats 5.6x, food 5.0x). The local
rule, at the same training fit, transfers almost nothing (10.0-10.8%, 1.23-1.31x). A
near-duplicate audit confirms the transfer is category structure, not eval-set leakage.
Mechanistically, the rules store the same training solution differently: the local rule
memorizes the pairing almost entirely in its shallowest branch code (train retrieval 0.997 at
tap 0; deeper taps at chance), while backprop distributes it across depth -- and only the
distributed solution carries transferable structure. We further document the optimization
regime needed to make the comparison fair (the local rule's coupling washes out unless the
contrastive term stays on through joint training, and destabilizes at standard rates), with a
full tuning sweep. The results identify a credit-assignment gap: a local rule can satisfy a
contrastive objective by arranging its latent geometry on the training set (alignment) without
propagating pair-discriminative pressure into its encoders (binding).

## 1. Setup

- Architecture (identical in both arms, byte-matched init and split): 4-stage conv image
  encoder with 4 tap codes; 4-block transformer text encoder with 4 tap codes; shared latent =
  l2-normalized concatenation of tap codes; linear decoders. 156.7M params, 64px, char-level
  captions, one caption per image.
- Data: COCO train2017 cache, N_train=20,000 pairs, N_eval=2,000 held-out pairs, split by
  image (no caption leakage). Vocab built on train only.
- BP arm (E1L): plain LARS + symmetric InfoNCE (temp 0.07), early-stopped at train latent
  retrieval >= 0.95 (fit gate). Wall ~7 min/seed.
- PC arm: relax-then-step local learning on a single energy F (reconstruction + cross
  prediction), with the InfoNCE coupling error injected at the deepest codes during relaxation
  and maintained through joint training (jointw=1.0). Stability recipe: lr 5e-3, warm-up 6000,
  150 epochs (see Sec. 4). Train latent retrieval 0.997-0.999 -- the local rule fits the
  training coupling as well as backprop.
- Decisive comparison: HELD-OUT transfer at matched training fit.

## 2. Headline result: category-level transfer dissociates the rules

Held-out image->caption category precision@10 (keyword-derived categories, ~1,720 categorized
eval items/seed; base rate = category frequency in the eval pool):

| seed | BP prec@10 (lift) | PC prec@10 (lift) | BP instance hits/2000 | PC hits/2000 | train fit BP / PC |
|---|---|---|---|---|---|
| 0 | 0.2146 (2.64x) | 0.0998 (1.23x) | 18 (17.0 sigma) | 3 | 0.981 / 0.997 |
| 1 | 0.2001 (2.43x) | 0.1080 (1.31x) | 12 (11.0 sigma) | 7 | 0.951 / 0.998 |
| 2 | 0.1995 (2.47x) | 0.1023 (1.27x) | 11 (10.0 sigma) | 2 | 0.972 / 0.999 |

Non-overlapping ranges across seeds. BP's lift is systematic per category (seed 0: elephant
7.2x, plane 6.1x, cat 5.6x, food 5.0x, train 4.3x, sports 3.8x, bathroom 3.2x); PC is flat
(~1x) essentially everywhere. Instance-level top-1-in-2000 shows the same direction (BP 11-18
vs PC 2-7) but is fragile (cache- and scale-sensitive; Sec. 5) -- the category metric is the
stable phenomenon underneath.

## 3. The transfer is category structure, not leakage (dupe audit)

COCO contains near-duplicate scenes; on this cache the eval pool could contain near-dupes of
training images, so held-out "hits" could be memorization leakage. Audit: recompute all
held-out hits (union i2t+t2i = 35/2000, chance ~1-2) and measure each eval item's nearest
TRAIN image (pixel L2). Hits' nearest-train per-pixel RMSE ~0.17 (median 0.169; true dupes are
<0.05), and hit-vs-nonhit medians differ only mildly (0.169 vs 0.202); most hits' nearest train
images are unrelated scenes (a motorcycle racer's nearest is a sheep field). The hits instead
cluster in frequent categories (planes-in-sky, bathrooms, field animals) -- category binding,
not instance leakage. The mild closest-decile enrichment (3-4x) is what category clustering
predicts (sky images resemble each other).

## 4. Making the comparison fair: what the local rule needs (and where it breaks)

A naive PC arm produces a confounded comparison, and we document the full path (appendix
sweep):
- Coupling OFF during joint training (jointw=0, the natural "warm-up then recon" schedule):
  the training coupling itself washes out with scale -- train latent retrieval 0.015-0.023 at
  2k but ~chance at 8k/20k. Any held-out failure is then a fitting failure, not a
  generalization failure.
- Coupling maintained (jointw sweep at 8k, 45ep): 0.1 -> train 0.0013 (washes), 0.3 -> 0.0107
  (partial), 1.0 -> 0.996 (fits, matching BP). Held-out stays at chance throughout.
- At 20k, jointw=1.0 at the recipe rate (2e-2) DIVERGES mid-training (weight movement 5,800%);
  lr 5e-3 trains stably to fit 0.997-0.999. (BP has its own edge: plain LARS collapses to a
  dead model at 40-80k pairs at its standard rate.)
The reported PC arm is therefore the strongest stable configuration we found: coupling
maintained at full weight, at its stable rate, fit-matched to BP. The dissociation of Sec. 2
is measured there, not at a strawman setting.

## 5. Mechanism: where each rule stores the training solution

Per-tap probes on the trained checkpoints (matched fit):
- The local rule memorizes the pairing almost entirely in its SHALLOWEST tap: train top-1
  retrieval 0.997 at tap 0; taps 1-3 at 0.02/0.01/0.003. Backprop distributes it: 0.96 / 0.94
  / 0.71 across taps 0-2.
- Held-out, all taps of both models are at chance on instance retrieval at 8k; the transferable
  category structure (Sec. 2) appears in BP's codes at 20k while PC's remain flat.
- During PC training with coupling on, global alignment collapses (held-out align_cos 0.95 ->
  0.05 as jointw rises to fitting strength) while per-pair discriminability never leaves
  chance: the local rule buys training-set alignment by arranging latent geometry, not by
  extracting pair-discriminative features -- alignment without binding.
Supporting evidence from the same system's generative arm: InfoNCE coupling drives batch
retrieval accuracy to 1.0 (identity alignment) without the text code approaching the decodable
image-latent value (value binding) -- the caption-to-image decode stays at a template across
six alignment routes (closed-form ridge, three distillation variants, InfoNCE at two batch
sizes).

## 6. Related work (to expand)

PC/local-learning at scale (Millidge, Salvatori, Song, Alonso et al.); BP-vs-PC equivalence
results and their fixed-prediction assumptions; contrastive representation learning and
alignment/uniformity analyses (Wang & Isola); CLIP-scale cross-modal learning; critiques of
retrieval-pool metrics.

## 7. Limitations (stated plainly)

Single model scale (156M) and one architecture family; 20k-pair from-scratch regime (four
orders below CLIP scale -- the claim is about the RULES' relative behavior at matched fit, not
about absolute capability); keyword-derived categories (crude but conservative); instance-level
effects are small and configuration-sensitive (documented); both rules exhibit scale-dependent
training instabilities (documented); PC arm is one local-learning instantiation, not all.

## Appendix A. Source files for every number (all committed)

| claim | source |
|:--|:--|
| Category table (headline) | catprobe_9431.log (s0), catall_9438.log (s1,s2), tools/category_probe.py |
| BP 20k seeds | E1L_results.json (11/13/13), cs_gate/e1l_20k_audit runs (18, 12, 11 w/ ckpts) |
| PC 20k stable seeds | cs_gate/jw10_20k_lr5e-3{,_s1,_s2}/coupling_scale_results_seed{0,1,2}.json |
| jointw sweep 8k | cs_gate/jw{01,03,10}/coupling_scale_results_seed0.json |
| jointw=0 washout at scale | res_8k*.json, res_20k*.json (train lat_retr ~chance) |
| PC 20k divergence at 2e-2 | cs_gate/jw10_20k_long_s0 run log (step 22,328) |
| BP collapse at 40-80k lr 1e-2 | cs_gate/e1l_{40000,80000} run logs |
| Dupe audit | cs_gate/e1l_20k_audit/audit2_*.log + dupe_audit.png, tools/dupe_audit.py |
| Per-tap anatomy | mech_9402.log, tools/mechanism_probe.py |
| Identity-vs-value (generation arm) | docs/experiments/LOG.md 2026-07-19..21 entries + tools/* |
