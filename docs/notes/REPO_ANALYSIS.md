# encoder_encoder_PCN - Repository Analysis

A read-and-explain writeup of what this repo is, how it is built, whether the design is sound, and whether it is worth more time. No code was changed and nothing was trained to produce this. It builds on what we already established (the OOM is just the model's size, and training diverges structurally because the state update and the weight update do not descend a single shared energy).

Author of the original repo is Anthony Yeh (Colby College). This analysis is of the fork at github.com/25simsa1/encoder_encoder_PCN.

---

## 1. Purpose - what is this model meant to do

**Short version.** It is meant to be a multimodal predictive-coding network that learns a shared latent representation of images and their captions, so that it can convert between the two in both directions. Give it an image, relax the network, and read out a caption. Give it a caption, relax, and read out an image. The README calls this "Bidirectional Predictive Coding for Multimodal Image-Text Conversion."

**Evidence.**
- `README.md` states the goal directly. Image goes to shared latent states goes to text reconstruction, and text goes to shared latent states goes to image reconstruction, with "iterative refinement via predictive error minimization" and shared states acting as a "shared memory substrate across modalities." It names "nocaps-style multimodal data."
- `data_preprocessing.ipynb` shows the actual data is **COCO 2017** (it downloads `train2017.zip` and `annotations_trainval2017.zip`). Images are read with `cv2.imread` and resized to 572x572. Captions are turned into **character-level one-hot vectors** (52 distinct characters, padded or cut to 192 positions), with a `-1e9` attention mask marking the `'\0'` padding. Only about 100 images and 501 captions are used.
- The same notebook demonstrates the intended use as image-to-caption. It calls `eepcn.test_step(1, image, zeros_like(text), predict='txt')`, which clamps the image, zeros the text, relaxes the network, and reads the text back out. For an image captioned "Rows of motor bikes and helmets in a city" the model produced `"ooooo..."` (the same character repeated). That run was on an untrained model, so the garbage output is expected, but it is concrete evidence the model was never trained to do the task.

**So the inputs and outputs mean.** The image branch input `(1, 572, 572, 3)` is one COCO photo (572 is indeed the classic U-Net input size, which is almost certainly where the number came from). The text branch input is `(1, 192, 52)` in the real pipeline, which is one caption as 192 character slots of 52-way one-hot, not the `(1, 192, 512)` you see in `repro_encoder_pcn.py` (that file is a synthetic smoke test with placeholder shapes). The `512` that shows up in the model is the embedding width after the first text Dense layer, not the input width.

**Honesty note.** The high-level intent is stated in the README, so I am not guessing about "image-caption, bidirectional." What is not written down anywhere is a precise training objective or a success metric, and as we found, the pieces do not actually add up to one. So the *stated* purpose is clear, the *worked-out* purpose (a concrete loss that the updates minimize) does not exist.

---

## 2. What the architecture actually is

Two encoders feed a stack of reconstruction "heads," and the heads of the two branches are tied together at five places so the modalities share latent states. Here is the data flow.

### Image branch (a VGG-style conv stack)
`encoder_encoder_pcn.py` builds, from `img_input` (clamped):
conv1(64) - conv2(64) - maxpool - conv3(128) - conv4(128) - maxpool - conv5(256) - conv6(256) - maxpool - conv7(512) - conv8(512) - maxpool - conv9(1024).
Every conv is 3x3 VALID (so each conv shrinks H and W by 2) and each maxpool halves them, so 572x572 collapses to 30x30x1024. This is a recognizable VGG-like downsampling tower.

### Text branch (a transformer pyramid with sequence resizing)
From `txt_input` (clamped): a Dense embedding to width 512, a sinusoidal positional encoding, then three `TransformerPCNLayer` blocks at width 512. Then a curious move. A Dense to 1024, a transpose, a Dense to 48, a transpose. That Dense-to-48 acts on the transposed tensor, so it **shrinks the sequence length from 192 to 48** while a parallel set of linears grows the feature width. This repeats. The sequence goes 192 then 48 then 12 then 3, while the width goes 512 then 1024 then 2048 then 4096, ending in eight transformer blocks at width 4096 operating over only 3 positions. So the text is squeezed into a very wide, very short representation.

### The reconstruction heads (where the parameters and the trouble live)
Off several depths of each branch, the model hangs a head with this shape.
`Flatten(feature map) - Dense(100) - Dense(big) - Dense(100) - Dense(big2)`.
On the image side the heads come off conv9, conv8, conv6, conv4, and conv2. On the text side they come off the last and several intermediate transformer blocks. In predictive-coding terms each `Dense(100)` is a 100-unit latent "state," and each head is trying to predict (reconstruct) the feature map it hangs off of, through that 100-unit bottleneck.

### How the two branches relate (the actual point of the model)
They are not parallel and independent. The text-side heads are constructed with `share_state_layer=` pointing at the matching image-side head, in five places (`dense4` shares state with `dense2`, `dense8` with `dense6`, `dense12` with `dense10`, `dense16` with `dense14`, `dense20` with `dense18`). "Shares state" means the two layers literally use the same state `tf.Variable`. So at five scales there is one shared latent that the image branch drives from one side and the text branch drives from the other. Making both branches agree on those shared states is the entire mechanism for "bidirectional image-text." That part of the idea is actually coherent.

### In predictive-coding language
- **Clamped** are the two inputs (`img_input`, `txt_input`). During `train_step` both are held fixed.
- **Latent states** are everything else - the conv and transformer activations and the `Dense(100)` bottlenecks - stored as `state` `tf.Variable`s and updated by `update_state`.
- **What each head predicts** is the feature map below it, via `predict_prev` (top-down reconstruction), with the prediction error being state minus reconstruction.
- **Relaxation then learning.** `update_states_wts_b` loops over layers calling `update_state` (settle the latents), then `update_wts` and `update_b` (adjust the weights to reduce the errors). `test_step` is the inference mode. It clamps one modality, frees the other, relaxes the states, and reads the freed modality back out. That is how a trained version would caption an image or imagine an image from a caption.

---

## 3. Is the design sound or idiosyncratic

Honest answer. The core idea (shared-latent multimodal predictive coding) is legitimate and interesting, but the concrete realization is idiosyncratic in several ways, one of them fatal.

### 3a. The flatten-into-Dense reconstruction heads are the central design flaw
A head does `Flatten(conv map) - Dense(100)`. For the conv2 head the conv map is 568x568x64 which is about 20.6 million numbers, so that single Dense weight matrix is 20.6M x 100, roughly 2.06 billion parameters in one layer. Summed across the five image heads this is the bulk of the ~7.7B total. This is not a sensible way to build a bottleneck. Flattening a full-resolution spatial map throws away the spatial structure that the convolutions just built, and a 100-unit code does not need a 2-billion-parameter projection to be reached. It is the thing that makes the model enormous for essentially no representational benefit.

What a normal design does instead. Reduce spatially before going dense. Global average pooling over the conv map (giving 64 or 1024 numbers, not 20 million), or a 1x1 conv, or a few strided convs, or an attention pooling token. Any of these would cut the head from billions of parameters to thousands or millions and would preserve, rather than destroy, the spatial information. A U-Net or autoencoder would also reconstruct through transposed convs, not through a giant dense layer. So this is the single highest-impact thing to change, and it is squarely "idiosyncratic," not standard.

### 3b. The attention scaling is non-standard
In `transformer_pcn_layer.py` the attention is `(q @ k_transposed) / (d_model // num_heads)`, i.e. it divides by d_k. The standard transformer (Vaswani et al. 2017, "Attention Is All You Need") divides by sqrt(d_k). Dividing by d_k instead of sqrt(d_k) over-shrinks the logits, which flattens the softmax toward uniform and makes attention barely distinguish positions, especially at the wide blocks where d_k is 512. This is a real departure, though it is a minor problem next to 3a and 3d.

### 3c. The data choices are unusual and make the task harder than it needs to be
Character-level one-hot text (52 characters, 192 positions) is a hard way to do captioning. Almost all image-captioning work uses word or subword tokens, because character-level generation is much harder to learn. And the images are fed as raw `cv2` BGR pixels in the 0-255 range with no normalization, which is bad for any neural network and would alone hinder training. The dataset is also tiny (about 100 images), which is far too little for a 7.7B-parameter model.

### 3d. The predictive-coding updates are PCN-flavored but not a faithful, single-objective formulation - this is the fatal one
The shapes are right. `predict_prev` is a top-down prediction, `predict_next` returns the state, errors are state minus prediction, and the weight updates are local and Hebbian-looking. Anyone who knows predictive coding will recognize the family. The classic references for this family are Rao and Ballard 1999 (predictive coding in visual cortex), Friston's free-energy work (2005 onward), and the modern machine-learning treatments by Bogacz 2017 (a tutorial with the explicit equations), Whittington and Bogacz 2017 (PCN with local Hebbian plasticity), and Millidge, Tschantz and Buckley 2022 (predictive coding approximates backprop).

In all of those, the inference (state) update and the learning (weight) update are both gradient descent on the **same** scalar free energy F, which is a sum of precision-weighted squared prediction errors. That shared F is what guarantees training drives the errors down. The state step is minus dF/dstate and the weight step is minus dF/dweights, by construction.

This repo does not have that property, and that is exactly what we proved empirically. The updates here are hand-derived per layer with several inconsistencies. The relu layers multiply the error by a GELU derivative (`d_gelu`) rather than the ReLU derivative, the state update mixes a "predict from next" term and a "predict from previous" term with ad hoc averaging, and the bidirectional terms are not the partials of one energy. We tested fixes (correcting the ReLU derivative, heavy relaxation, learning-rate decoupling, gradient normalization, state clipping) and the prediction-error energy still climbed about four orders of magnitude during training instead of dropping. So the divergence is not a one-line slip in an otherwise sound objective. It is a structural mismatch. There is no single F that these updates jointly descend, so there is no fixed point for training to find, and the only reason the run stays numerically finite is the state clip we added, which pins a runaway state rather than letting the dynamics settle.

Conclusion for section 3. The headline idea is sound and has real precedent. The realization departs from standard practice in the heads (3a), the attention (3b), and the data (3c), and most importantly the learning rule is not derived from a single energy (3d), which is why it cannot train.

---

## 4. Does anything work

Separating "runs" from "does something useful."

- **Runs without crashing.** Yes, with enough VRAM and the async allocator, one `train_step` and `test_step` complete. We confirmed this. But "runs" here means "executes the math without erroring," not "learns."
- **Trains.** No. The energy climbs, the author never finished (last commit message is literally "still need to fix oom"), the data notebook's training loop crashed before completing, and the only readout we have is the degenerate `"ooooo"` caption from an untrained model.
- **Parts with genuine value.**
  - The conceptual design (a shared latent that two modalities both reconstruct into, for bidirectional conversion) is a nice idea and is the salvageable intellectual core.
  - The layer abstractions are reasonable scaffolding. Each layer exposing `predict_prev`, `predict_next`, `update_state`, `update_wts`, `update_b`, and a relaxation loop that iterates them, is the right shape for a PCN even though the specific update math is wrong.
  - The data pipeline runs mechanically (download COCO, resize, tokenize, mask), so it is reusable as a starting point, though it needs normalization and ideally word or subword tokens.
- **Parts with no value as written.** The giant flatten-into-Dense heads, the specific hand-derived update equations, and the 7.7B parameter count. These should not be carried forward.

Net assessment. It is a non-functional research prototype. The idea is interesting, the execution does not train, and the most expensive parts of the code are the least defensible.

---

## 5. Bottom line - three direct answers

### 5a. Fix this repo, or start from a maintained predictive-coding library

Start from a maintained library. Do not try to make this one train. Three real, currently-maintained options, with what each is for.

- **PCX (`pcx`)**, JAX, github.com/liukidar/pcx. From the Bogacz and Salvatori and Lukasiewicz group, released with "Benchmarking Predictive Coding Networks - Made Simple" (Pinchetti et al. 2024). This is the best fit if you want to do predictive-coding research with sound, single-energy objectives and ready-made benchmarks (MNIST, CIFAR, autoencoding, associative memory). It is fast (JAX JIT) and is the closest thing to a reference modern PCN library.
- **ngc-learn**, JAX, github.com/NACLab/ngc-learn (NAC lab, RIT). A broader neuro-AI and computational-neuroscience toolkit that includes predictive coding circuits, with a companion model zoo (ngc-museum). Best fit if you want biologically-motivated predictive processing more generally, not only classification benchmarks.
- **predify**, PyTorch, github.com/miladmozafari/predify. Converts an existing PyTorch CNN into a predictive network by adding predictive-coding feedback dynamics via a config file. Best fit if your interest is specifically vision and you want to bolt predictive coding onto a standard CNN quickly.

One clarification, since you mentioned pyhgf. pyhgf implements the Hierarchical Gaussian Filter, which is Bayesian filtering for continuous, often time-series signals (volatility, reinforcement learning, neuroscience). It is predictive-processing-adjacent but it is not a general image-and-text PCN, so it does not fit this use case. (Other PyTorch options if you prefer that ecosystem are PRECO and Torch2PC.)

### 5b. The minimal sound thing to build if the goal is to learn PCN by training one

Build a small predictive-coding network that actually descends one energy, and build it clean rather than salvaging this. Concretely, a 2 or 3 layer fully-connected PCN on MNIST or Fashion-MNIST, following the explicit equations in Bogacz 2017 or Whittington and Bogacz 2017, where the state update is minus the gradient of F and the weight update is minus the gradient of the same F. That is about 100 lines, it trains in minutes on a laptop, and you will see the energy go down, which is the whole lesson. PCX ships almost exactly this as an example, so the fastest path is to run the PCX MNIST example, confirm the energy drops, then modify it.

Is it salvageable from this code. Only the idea, not the code. The update rules are the part you most need to be correct, and they are the part that is wrong, so reusing them defeats the purpose of the exercise.

### 5c. Is this repo worth any more of your time

No. Direct reasoning, not a hedge.

- It was never trained end to end, and the author abandoned it at "still need to fix oom."
- The failure to train is structural, not a tuning problem. The state and weight updates do not share an energy, which we confirmed across multiple fixes. Repairing that means re-deriving the entire learning rule, at which point you have rewritten the model.
- The architecture is 7.7B parameters mostly because of the flatten-into-Dense heads, which are a design mistake, so even a correct learning rule would be training a needlessly enormous and poorly-conditioned model on about 100 unnormalized images with character-level captions.
- Mature, fast, correct PCN libraries already exist (5a), so there is no tooling gap this repo fills.

The one thing worth keeping is the concept. A shared-latent multimodal predictive-coding model for image-text conversion is a genuinely interesting thesis-sized idea. The right move is to take that idea to PCX or ngc-learn, start small (single modality, a sound energy, a real benchmark), confirm it trains, and only then add the second modality and the shared latents. Keep the idea, drop the code.

---

## Sources

Predictive-coding literature referenced above.
- Rao and Ballard, 1999, "Predictive coding in the visual cortex," Nature Neuroscience.
- Friston, 2005, "A theory of cortical responses," Phil. Trans. R. Soc. B (free-energy principle).
- Bogacz, 2017, "A tutorial on the free-energy framework for modelling perception and learning," J. Math. Psychology (has the explicit PCN update equations).
- Whittington and Bogacz, 2017, "An approximation of error backpropagation in a predictive coding network with local Hebbian plasticity," Neural Computation.
- Millidge, Tschantz and Buckley, 2022, "Predictive coding approximates backprop along arbitrary computation graphs," Neural Computation.
- Vaswani et al., 2017, "Attention Is All You Need" (for the standard 1/sqrt(d_k) attention scaling).

Maintained predictive-coding libraries.
- [PCX (pcx) on GitHub](https://github.com/liukidar/pcx) and the paper [Benchmarking Predictive Coding Networks - Made Simple (arXiv:2407.01163)](https://arxiv.org/abs/2407.01163v1)
- [ngc-learn on GitHub](https://github.com/NACLab/ngc-learn) and [ngc-learn documentation](https://ngc-learn.readthedocs.io/en/latest/)
- [predify on GitHub](https://github.com/miladmozafari/predify) and the paper [Predify (arXiv:2106.02749)](https://arxiv.org/pdf/2106.02749)
- Other PyTorch options - [PRECO](https://github.com/bjornvz/PRECO), [Torch2PC](https://github.com/RobertRosenbaum/Torch2PC)
