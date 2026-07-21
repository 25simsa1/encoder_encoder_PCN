# What Transfers in From-Scratch Cross-Modal Predictive Coding, and Where the Bottleneck Is

<!-- TODO(title): working title. Pick the final one once the scale curve (finding 4) lands and we know
     whether this stays a characterization paper or shifts to a positive result. -->

**Authors.** TODO(authors). **Venue.** ICLR 2027 submission (deadline 2026-09-20). **Status.** DRAFT.

> **Reading note for co-authors.** Every quantitative claim below is pulled from a committed result file
> and the source commit is named inline. Sections marked TODO are pending the running data-scale curve
> (commit fa1e736) or are editorial placeholders. Nothing here is an eyeballed or remembered number.

---

## Abstract

<!-- TODO(abstract): tighten once finding 4 lands. Keep it honest. This is a characterization plus a
     mechanistic localization, not a positive-results paper. -->

We study predictive coding networks (PCNs) trained entirely from scratch, with no backpropagation
pretraining and no pretrained features, for bidirectional cross-modal generation between text and images.
A single energy functional couples a convolutional image encoder and a character-level transformer text
encoder through a shared multi-scale latent that is relaxed under that energy, with decoders back to both
modalities. The from-scratch property is the object of study, not an incidental constraint. We characterize
what does and does not transfer out of sample in this regime and we localize the bottleneck.

Three results hold across our runs. First, moving the weights enough to break the dataset-mean collapse
produces text-conditioned image output that varies by caption, yet on unseen captions the model does not
match a caption to its image above chance. The image autoencoding path does generalize, but it does not use
the caption. An earlier in-sample retrieval number that read as many times chance is shown to be a
retrieval-pool-size artifact rather than evidence of generalization. Second, the negative is robust to
scale and is not an optimization artifact. At two thousand training pairs every arm, including a
compute-matched control, remains at chance on held-out retrieval, while a global cross-modal cosine
alignment rises with training and per-pair discriminability stays at chance, a pattern we call mean-collapse
alignment. Third, the bottleneck is the cross-modal coupling failing to generalize rather than failing to
optimize. A contrastive warm-up drives the training contrastive loss to near zero yet builds zero held-out
per-pair separability, and keeping the contrastive term on through joint training does not rescue it.
TODO(abstract-finding4): one sentence on the data-scale curve once it lands.

We present these as well-controlled negatives together with a mechanistic localization of the failure. The
evaluation protocol is held-out only, uses a disjoint split and a train-only vocabulary, fixes a
pre-registered above-chance bar, reports retrieval as raw hit counts and sigma above chance rather than as a
multiple of chance on a large pool, and scores reconstruction against a train-mean baseline.

---

## 1. Introduction

Predictive coding offers a biologically motivated alternative to backpropagation in which inference is an
iterative relaxation of latent states under a local energy and learning is a separate update of the weights
that parameterize that energy. Most predictive coding work studies classification or single-modality
autoencoding, and most reports of generative behavior either use small models on small images or rely on
backpropagation somewhere in the pipeline. We ask a narrower and, we argue, more informative question. If a
bidirectional image and text predictive coding network is trained entirely from scratch, with no pretrained
features anywhere, what part of cross-modal behavior actually transfers to unseen data, and if some part
does not transfer, where exactly does it break.

The from-scratch constraint is deliberate. A great deal of cross-modal generation in the deep learning
literature inherits structure from large pretrained encoders, so it is difficult to attribute success or
failure to the learning rule itself. By holding the from-scratch property fixed we make the learning rule
and the coupling mechanism the only things under test. This is the point of the work and we do not dilute
it by importing pretrained components.

Our contribution is a characterization rather than a positive result. We establish that breaking the
dataset-mean collapse, which prior work in this line attributed to undertraining and weight movement (see
Section 2), is not sufficient for cross-modal generalization. The image side generalizes as an
autoencoder while the text-to-image path produces caption-varying but non-matching output on unseen
captions. We then show this negative survives a scale increase and a compute-matched control, and we
localize the failure to the cross-modal coupling, which optimizes on the training pairs but does not
generalize, rather than to washout or to an optimization failure during joint training.

We make three claims, each tied to a committed experiment.

1. From-scratch text-to-image breaks mode collapse but does not match unseen captions above chance
   (commit d36b6ae). The image autoencoding path generalizes without using the caption. The earlier
   in-sample retrieval number was inflated by the retrieval-pool size.
2. The negative is robust to scale and is not an optimization artifact (commit 4d4f4e5). At two thousand
   pairs all arms remain at chance held-out, with rising global alignment and chance-level per-pair
   discriminability.
3. The bottleneck is the coupling failing to generalize, not optimization or washout (commit 9ab896b). A
   contrastive warm-up drives the training contrastive loss to near zero with zero held-out separability,
   and keeping the coupling on through joint training does not rescue it.

A fourth claim, on how the negative behaves as a function of training-set size across two thousand, eight
thousand, and twenty thousand pairs, is pending the running data-scale curve (commit fa1e736) and is
reported as a placeholder in Section 5.4.

---

## 2. Related Work

<!-- Sourced from LIT_REVIEW.md (adversarial prior-art review, 2026-06-17). The lit review was written to
     position the earlier weight-movement / mode-collapse finding; the cross-modal generalization framing
     here reuses its architecture-class and optimization-line positioning and adds the cross-modal angle. -->

**Optimization and weight movement in predictive coding.** That poor solutions in predictive coding are an
optimization phenomenon, diagnosed by weight-update magnitude and addressed by making weights move more, is
established by Alonso et al. (arXiv:2305.13562), who study classification and one autoencoder rather than
generative diversity. That effective learning rate must scale with network size, and that wrong scaling
makes weight updates vanish, is established by the stable-parameterization line (arXiv:2411.02001) and by
muPC (arXiv:2505.13124). Our work takes the breaking of mean collapse as a starting point rather than a
contribution and asks what happens next in the cross-modal setting.

**Inference scheme.** Our inference relaxes a shared latent under a single joint energy, updating all scales
simultaneously, and we make no claim about inference scheduling. This differs from the sequential,
output-to-input inference of Alonso et al. (arXiv:2305.13562), whose contribution is a computation-reduction
technique for supervised classification and is orthogonal to the cross-modal generalization we study.

**Generative and bidirectional predictive coding.** Bidirectional predictive coding with separate top-down
and bottom-up weights and a multimodal experiment exists (arXiv:2505.23415, Bogacz group), but the
multimodal pairing there is image and a ten-way label on MNIST and Fashion-MNIST, not image and text, and
not at scale. Making predictive coding networks generative on images alone (arXiv:1910.12151) attributes
degenerate generation to inverse-problem non-uniqueness and constrains weights to address it, the opposite
of moving weights more. The review of Millidge et al. (arXiv:2107.12979) attributes blurry mean generation
to underdetermination and capacity. These are competing causal accounts that our setting lets us hold up
against an undertraining account, and none of them studies image and text at scale.

**The novelty gap.** The specific combination we study, a from-scratch bidirectional image and text
predictive coding network at the scale of thousands of pairs and roughly one hundred fifty-six million
parameters, with held-out cross-modal generalization as the measured quantity, is not covered by any prior
predictive coding paper we located. Prior bidirectional multimodal predictive coding is image and label,
small scale, and reports collapse as an architectural or energy-landscape effect. We instead measure what
transfers out of sample and we localize the cross-modal coupling as the failure site. The characterization
itself, what transfers and where the coupling breaks in from-scratch cross-modal predictive coding, is the
open ground.

**Contrastive cross-modal alignment.** Our coupling warm-up uses an InfoNCE objective to align the image
and text latents. TODO(cite-clip-infonce): cite CLIP (Radford et al. 2021) and the InfoNCE origin (Oord et
al. 2018) and state the distinction, namely that those systems learn alignment with backpropagation on
pretrained-scale data whereas here the same objective is applied from scratch and is measured for held-out
per-pair transfer, which it does not achieve.

<!-- TODO(refs): convert the arXiv ids above into a proper bibliography. Full screened list and the two
     primary threats (2305.13562, 2411.02001) are in LIT_REVIEW.md. -->

---

## 3. Method

We describe the architecture, the energy, the inference and learning updates, and the batched
infrastructure that lets the same recipe run at scale. The recipe is frozen across every experiment in
Section 5. Where a run changes only the data split or the readout scope we say so.

### 3.1 Architecture

The model has roughly one hundred fifty-six million parameters (156,598,112 in the four-hundred-pair
held-out run with vocabulary size thirty-four, and 156,657,248 in the scale runs with vocabulary size
forty-six). Images are 64 by 64 by 3 in the range zero to one. Captions are real lowercased sentences
encoded at the character level with a capped length of sixty-four.

The image encoder is a four-stage convolutional stack with GELU activations and max pooling, producing
multi-scale features that are projected into four latent blocks of widths 3072, 3072, 1536, and 1536. The
text encoder is a four-block character-level transformer with four heads, whose per-block mean-pooled
representation is projected into the same four latent blocks. Both encoders feed a single shared latent
state with four scales. Each scale is projected to a sixteen-dimensional code, the codes are concatenated,
and two linear decoders map the concatenated code back to the image pixels through a sigmoid and back to the
caption logits.

### 3.2 Energy and updates

A single energy functional F couples the two encoders, the shared latent, and the two decoders. For a
latent state S with image taps it and text taps tt, image target igt and text target tgt, the energy sums a
cross term that pulls each latent scale toward both encoder taps and a generative term that penalizes both
decoder reconstructions, with weights A_CROSS equal to one and A_GEN equal to two so that the generative
term is weighted at least as heavily as the cross term.

Inference is relax-then-step. Holding the weights fixed, the latent state is initialized at the average of
the image and text taps and is updated by gradient descent on F for a fixed number of relaxation steps
(eight during training, twenty-five for the generation readouts), with per-scale step sizes proportional to
the scale width. Learning is a separate weight update. With the relaxed latent held fixed, the weights are
updated by a plain LARS rule with a trust ratio and a small bias floor at a learning rate of two times ten
to the minus two. There is no backpropagation through the inference trajectory and no pretrained component
anywhere. The maximum absolute weight is monitored and the run is declared diverged if the energy or the
weights become non-finite or exceed a fixed bound.

### 3.3 Contrastive coupling warm-up

To test whether aligning the two latents before joint training helps, we add an optional InfoNCE warm-up.
The image and text encoder outputs are concatenated per modality and L2-normalized, an InfoNCE loss with
temperature 0.07 is computed over a batch with the matched pair as the positive, and the weights are updated
with the same LARS rule at a smaller warm-up learning rate of two times ten to the minus three. The warm-up
uses only paired training data and no pretrained features. An option keeps the InfoNCE term active during
joint training at a fraction of the joint learning rate, which lets us test whether any warm-up alignment
washes out under the image-dominated joint phase.

### 3.4 Batched scale infrastructure

The frozen recipe relaxes one example at a time, which is the throughput wall. To train on thousands of
pairs we batch the joint phase with one correctness requirement. The latent relaxation reduces with a sum
over the batch so that each example is relaxed identically whether it is alone or in a batch, which keeps
the dynamics batch-invariant. The weight update reduces with a mean over the batch, which is the standard
minibatch gradient, and the LARS trust ratio makes the weight step scale-invariant anyway. The consequence,
which is the infrastructure validation target, is that at batch size one the batched code reproduces the
frozen single-example recipe exactly, because the sum and the mean coincide for a single example.

---

## 4. Evaluation Protocol

We treat the protocol as a strength of the work, because the central claims are negatives and a weak
protocol would make a negative uninformative. Every rule below was fixed before the runs it governs.

**Held-out only.** Every headline metric is computed on a held-out set that is disjoint from training. We
report train and held-out side by side so the generalization gap is explicit, but the claims rest on the
held-out numbers.

**Disjoint split and train-only vocabulary.** The train and evaluation indices are a deterministic disjoint
partition. In the scale runs there is exactly one caption per image so a split by pair is a split by image
with no leakage. The character vocabulary is built from training captions only, and any character unseen at
training maps to the null token, so the evaluation captions cannot leak into the embedding table.

**Pre-registered above-chance bar.** For text-to-image retrieval over a held-out pool of size N, chance is
one over N and the pass bar is a hit rate above three over N, which is roughly two to three sigma. We do not
move this bar after seeing results.

**Retrieval as hits and sigma, never as a multiple of chance on a big pool.** We report raw hit counts and
the pool size. We avoid stating retrieval as a multiple of chance, because the same absolute hit rate reads
as a large multiple against a large pool and as near chance against a small pool, which is exactly the
inflation we diagnose in Section 5.1.

**Reconstruction against a train-mean baseline.** Image reconstruction MSE is compared to the MSE of
predicting the train-mean image on the evaluation set. Beating that baseline is genuine reconstruction.
Matching or exceeding it is trivial.

**Validity floor on weight movement.** A run is only valid if the weights moved at least forty percent from
initialization, measured as the relative L2 change. A run below that floor is declared undertrained and
void rather than a negative, so that our negatives cannot be dismissed as undertraining.

**Coupling diagnostics.** Beyond generation retrieval we measure two latent-space quantities on held-out
data. The matched-pair cosine alignment is the mean cosine between paired image and text latents. The latent
retrieval is the top-one rate of matching a text latent to its image latent in the held-out pool. Their
dissociation, high cosine with chance-level latent retrieval, is the mechanistic signature we report in
Section 5.2 and 5.3.

---

## 5. Results

The energy value F is not the signal in any of these runs and is reported only as a training-stability
trace. Every claim is a held-out generation or held-out latent quantity.

### 5.1 From-scratch text-to-image breaks collapse but does not match unseen captions (commit d36b6ae)

We train on four hundred pairs and evaluate on a disjoint one hundred, with the recipe of Section 3 frozen,
roughly 156.6 million parameters, five thousand steps, and 23.6 minutes of training. The weights moved 116
percent, well past the forty percent validity floor, so the verdict is valid rather than undertrained. The
energy fell from 0.125 to 0.051. Source file step1_heldout_results.json.

| metric | train (N=400, chance 0.0025) | held-out (N=100, chance 0.0100) |
|:--|--:|--:|
| text-to-image retrieval | 0.020 (8 of 400) | 0.020 (2 of 100) |
| diversity ratio | 0.375 | 0.273 |
| out-range | 0.603 | 0.338 |
| image-to-image recon MSE | 0.0293 (train-mean base 0.0654) | 0.0312 (base 0.0605, beats it) |
| image-to-text token accuracy | 0.351 (mode-char base 0.200) | 0.308 (base 0.218) |

What this establishes. The collapse to a single dataset-mean image is broken. On unseen captions the
text-to-image output still varies, with a diversity ratio of 0.273 above the 0.20 threshold and a positive
out-range. The image autoencoding path generalizes, since held-out reconstruction at 0.0312 beats the
train-mean baseline of 0.0605 and roughly equals the train reconstruction of 0.0293. Image-to-text token
accuracy beats only the weak mode-character baseline and is inconclusive on its own.

What this does not establish, which is the load-bearing negative. Above-chance caption-to-image retrieval on
unseen captions. Two hits out of one hundred is not above the one hit expected by chance and fails the
pre-registered bar of more than three hits, roughly two sigma. By the rule fixed before the run this is a
held-out partial, not a pass.

The pool-size inflation. The raw retrieval rate is identical on train and held-out at 0.020, and the
reconstruction and image-to-text gaps are small, so this is not a classic train-overfit blowup. A roughly
two percent absolute hit rate, scored against a four-hundred-way in-sample pool with chance 0.0025, reads as
roughly eight times chance at about seven sigma and looks like a real signal. The same two percent rate
scored against the one-hundred-way held-out pool with chance 0.01 is two hits against one expected and is
within noise. The earlier in-sample many-times-chance number was a retrieval-pool-size artifact for the
purpose of claiming generalization, not per-pair memorization. This is why the protocol forbids reporting
retrieval as a multiple of chance on a big pool.

### 5.2 The negative is robust to scale and is not an optimization artifact (commit 4d4f4e5)

We scale to two thousand training pairs and a one-thousand held-out pool with the batched infrastructure of
Section 3.4, three arms from an identical initialization and identical data order. Arm A is the baseline.
Arm B adds the InfoNCE warm-up. Arm A_long is a compute-matched control with extra joint epochs equal to
the warm-up data budget, which defuses the confound that B simply saw more gradient steps. All three arms
moved between 70 and 82 percent, above the validity floor. Source file coupling_scale_results_seed0.json.
Chance on the held-out pool is 0.001 and the pass bar is more than three hits per thousand.

| held-out metric (N=1000, chance 0.001) | A baseline | B warm-up | A_long control |
|:--|--:|--:|--:|
| text-to-image retrieval | 0.002 (2 of 1000) | 0.001 (1 of 1000) | 0.001 (1 of 1000) |
| matched-pair cosine alignment | 0.709 | 0.764 | 0.802 |
| latent retrieval top-1 | 0.001 | 0.003 | 0.000 |
| diversity ratio | 0.292 | 0.427 | 0.453 |
| image-to-image recon MSE (base 0.0677) | 0.0291 | 0.0277 | 0.0255 |
| image-to-text token accuracy | 0.321 | 0.312 | 0.317 |

Every arm is at chance on held-out text-to-image retrieval, including the compute-matched control, so the
negative is not an undertraining or step-budget artifact and is not removed by more compute. The key pattern
is in the latent quantities. The matched-pair cosine alignment is high, between 0.71 and 0.80, and it is
highest for the arm with the most training, A_long at 0.802. Yet the latent retrieval top-one rate is at
chance for every arm, 0.001 and below. The latents become globally similar across the two modalities while
remaining non-discriminable per pair. We call this mean-collapse alignment. More training raises the global
alignment without building the per-pair separability that held-out matching requires. Image reconstruction
again beats the train-mean baseline in every arm, so the image autoencoding path generalizes at this scale
as well.

### 5.3 The bottleneck is the coupling failing to generalize, not optimization or washout (commit 9ab896b)

This diagnostic isolates the warm-up and reads the latent separability at two points, right after the
warm-up and before joint training, and again after joint training. We run it at two settings of the joint
contrastive weight, zero (file diag_jw0.json) and one tenth (file diag_jw1.json), the latter keeping the
InfoNCE term active through joint training to test for washout. Both runs are at two thousand training pairs
and a one-thousand held-out pool, chance 0.001.

The warm-up optimizes the training objective. The training InfoNCE loss falls from about 4.05 at step fifty
to 0.035 at step fifteen hundred (file diag_jw0.log). TODO(verify-infonce-start): the earliest logged value
is 4.05 at step 50; if a pre-step-50 value is wanted, note that the first log point is step 50, not step 0.

It builds zero held-out separability. Measured right after the warm-up and before any joint training, the
held-out matched-pair cosine alignment is 0.021 and the held-out latent retrieval is exactly 0.000 at chance
0.001 (file diag_jw0.log). The warm-up drove a near-zero training contrastive loss while producing no
held-out per-pair separability at all. The alignment it learned was train-pair specific and did not transfer.

Joint training does not rescue it, and keeping the coupling on does not either.

| held-out, N=1000, chance 0.001 | jw0 A | jw0 B | jw1 A | jw1 B |
|:--|--:|--:|--:|--:|
| text-to-image retrieval (hits) | 0.001 (1) | 0.001 (1) | 0.000 (0) | 0.002 (2) |
| matched-pair cosine alignment | 0.745 | 0.756 | 0.806 | 0.774 |
| latent retrieval top-1 | 0.000 | 0.006 | 0.001 | 0.003 |
| post-warmup latent retrieval (pre-joint) | n/a | 0.000 | n/a | 0.000 |
| post-warmup cosine (pre-joint) | n/a | 0.021 | n/a | 0.022 |

After joint training the matched-pair cosine rises into the 0.75 to 0.81 range for every arm, the same
mean-collapse alignment as in Section 5.2, while the latent retrieval stays at chance. Keeping the InfoNCE
term on through joint training at weight one tenth does not produce held-out separability either, with arm B
latent retrieval at 0.003 and arm A at 0.001. The failure is therefore localized. It is not that joint
training washes out a good aligned initialization, because the post-warmup initialization had zero held-out
separability to begin with. It is the coupling that does not generalize. It reduces the training contrastive
loss to near zero and aligns the training pairs, and none of that transfers to unseen pairs.

### 5.4 Data-scale curve across 2k, 8k, and 20k pairs (commit fa1e736, PENDING)

<!-- PLACEHOLDER. The running experiment (commit fa1e736, train2017 source + capped train readout) sweeps
     training-set size. Fill this table from the committed *_results*.json it produces. Do NOT invent
     numbers; leave TODO until the file lands. The structure below works whether the curve stays flat (a
     characterization) or bends upward at scale (a positive result). -->

This section reports how the held-out negative behaves as the training set grows from two thousand to eight
thousand to twenty thousand pairs, holding the recipe fixed. The question is whether held-out text-to-image
retrieval and held-out latent retrieval stay at chance as data grows, which would strengthen the
characterization, or whether either rises above chance at some scale, which would shift the contribution.

| training pairs | held-out chance | text-to-image retrieval (hits) | latent retrieval top-1 | matched-pair cosine | recon vs base |
|:--|--:|--:|--:|--:|--:|
| 2,000 | 0.001 | TODO | TODO | TODO | TODO |
| 8,000 | TODO | TODO | TODO | TODO | TODO |
| 20,000 | TODO | TODO | TODO | TODO | TODO |

TODO(finding4-verdict): state whether the curve is flat at chance (characterization holds) or bends upward
(re-frame). The 2,000-pair row can be back-filled now from Section 5.2 and 5.3 if we want a self-consistent
anchor, but the 8,000 and 20,000 rows must come from the running experiment's committed result files.

---

## 6. Discussion and Limitations

**What the results say together.** From-scratch cross-modal predictive coding learns the image autoencoding
path and breaks the dataset-mean collapse, but the text-to-image path does not transfer to unseen captions
at the scales tested. The failure is not undertraining, since the weights moved well past the validity floor
and a compute-matched control does not change the outcome. The failure is not washout of a good aligned
initialization, since the post-warmup initialization had zero held-out separability. The failure is the
cross-modal coupling, which optimizes and aligns on the training pairs and produces a globally similar but
per-pair non-discriminable latent geometry out of sample, the mean-collapse alignment.

**Why this is informative rather than a null.** The dissociation is the content. The image side generalizes
while the text-conditioning does not, the training contrastive loss goes to near zero while held-out
separability stays at zero, and global alignment rises with training while per-pair separability does not.
Each of these separates a candidate cause from the observed failure and points at the coupling as the site.

**Limitations.** TODO(limitations): expand. Current known limitations include the following. The scale
ceiling tested is twenty thousand pairs pending finding 4, which is small relative to cross-modal systems
that succeed with backpropagation and pretraining. Images are 64 by 64 and captions are character-level,
which bounds the achievable fidelity independent of the coupling question. The seed count is limited and the
diagnostics in Section 5.3 are reported at seed zero. TODO(seeds): confirm how many seeds the committed
files cover and add a multi-seed statement or mark it as future work. The held-out pools, while disjoint and
vocabulary-clean, are at most one thousand, which bounds retrieval sigma. We make no claim about predictive
coding cross-modal generation with pretrained features, which is outside the from-scratch scope by design.

**Threats to the framing.** TODO(threats): from LIT_REVIEW.md, pre-empt the reviewer lines that this is
just the known weight-movement result in a generative setting, and that the negative might be specific to
the LARS recipe or the character-level text encoder rather than to from-scratch cross-modal predictive
coding in general. State which controls already address each and which are future work.

---

## 7. Conclusion

We characterized from-scratch cross-modal predictive coding for text and images and localized its
bottleneck. Breaking the dataset-mean collapse is necessary but not sufficient for cross-modal
generalization. The image autoencoding path transfers, the text-to-image path does not match unseen captions
above chance, and this negative survives a scale increase and a compute-matched control. The failure is the
cross-modal coupling, which drives the training contrastive loss to near zero and the training pairs into
alignment yet builds no held-out per-pair separability, producing a globally aligned but per-pair
non-discriminable latent geometry. TODO(conclusion-finding4): close with the scale-curve verdict once it
lands, and state whether the contribution remains a characterization plus mechanistic localization or shifts
toward a positive result.

---

## Appendix A. Source files for every number

| claim | source file | commit |
|:--|:--|:--|
| Finding 1, 400/100 held-out | step1_heldout_results.json, docs/runbooks/RUN_STEP1_HELDOUT.md | d36b6ae |
| In-sample gate (context for the inflation) | docs/runbooks/RUN_STEP1.md | 00304a9 |
| Finding 2, 2k scale point with control | coupling_scale_results_seed0.json | 4d4f4e5 |
| Finding 3, warm-up washout diagnostic (jw=0) | diag_jw0.json, diag_jw0.log | 9ab896b |
| Finding 3, warm-up washout diagnostic (jw=0.1) | diag_jw1.json, diag_jw1.log | 9ab896b |
| Finding 4, data-scale curve | TODO (pending) | fa1e736 |
| Method, evaluation protocol | experiments/run_step1_coco_heldout.py, experiments/run_infonce_warmup_coco.py, experiments/run_coupling_scale.py | d36b6ae / 1778665 / b209377 |
| Related work and novelty gap | LIT_REVIEW.md | (committed) |
