# PCN_FIX_PLAN.md - Re-deriving the update rules from one free energy

A research-and-planning document. No code was changed and nothing was trained to produce it. The goal is to fix the training divergence whose root cause we already established. The hand-written `update_state` and `update_wts` do not descend a single shared energy, so there is no fixed point and the prediction-error energy climbs. The only real fix is to re-derive both updates as gradients of one explicit free energy. Everything below is grounded in papers pulled directly from the arXiv API, with IDs and links in the Sources section.

The key takeaway up front. In a correct predictive-coding network the state update and the weight update are both gradient descent on the SAME scalar F, and both contain the SAME activation derivative f'. The repo violates exactly this in three concrete ways (a GELU derivative used for ReLU layers, an ad-hoc averaging-plus-extra-term structure in the state update, and a LARS rescaling of the weight update). Fix those and the energy will descend, because the literature proves F is non-increasing for the correct rules.

---

## 1. Canonical theory (the equations, from the arXiv PDFs)

There are two equivalent ways the literature writes predictive coding. The chain (layer) form and the arbitrary-computation-graph (DAG) form. This model is a DAG (two branches, reconstruction heads, shared latents), so the DAG form is the one to implement, but the chain form is easier to read first.

### 1a. Chain form (Millidge, Song, Salvatori, Lukasiewicz 2022, arXiv 2207.12316, Section 2)

A PCN is layers of activities `x_0 ... x_L`. Each layer makes a prediction of the next layer from the layer below,
- prediction  `x_hat_l = f(W_l x_{l-1})`
- error       `eps_l = x_l - f(W_l x_{l-1})`

The whole network has ONE energy (a sum of squared prediction errors),
- `F = sum over l of  eps_l^2`.

Both phases are gradient descent on that same F.
- Inference (the latent states), eq (1) of 2207.12316:
  `x_dot_l = -dF/dx_l = -eps_l + (eps_{l+1} . f'(W_{l+1} x_l)) W_{l+1}^T`   for l < L,
  `x_dot_l = -eps_l`                                                        for l = L.
- Learning (the weights), eq (2), evaluated at the inference equilibrium x*:
  `W_dot_l = -dF/dW_l = (eps_l . f'(W_l x*_{l-1})) x*_{l-1}^T`.

Here `.` is elementwise multiplication and `f'` is the derivative of the SAME activation `f` used in the forward prediction. Notice `f'` appears in BOTH the inference and the learning update. That shared, consistent `f'` is the crux. The paper proves (Section 3.4, Prop 3.1 and Thm 3.6) that this F equals the supervised loss plus a non-negative residual, so driving F down provably drives the task loss down, and a PCN trained this way converges to a critical point of the loss. A correct PCN's energy goes DOWN by construction. Ours goes up, which is the signature of broken rules.

### 1b. DAG form (Millidge, Tschantz, Buckley 2020/2022, arXiv 2006.04182, Section 2)

For an arbitrary directed acyclic computation graph with vertices `v_i`, each vertex is predicted from its parents,
- prediction  `v_hat_i = f(parents of v_i ; theta_i)`
- error       `eps_i = v_i - v_hat_i`
- energy      `F = sum over all vertices i of  eps_i^2`   (eq 1)

Updates (eq 2 and eq 3, with the signs as drawn in their Figure 1),
- Inference  `v_dot_i = -dF/dv_i = -eps_i + sum over children j of  eps_j (d v_hat_j / d v_i)`
- Learning   `theta_dot_i = -dF/dtheta_i = eps_i (d v_hat_i / d theta_i)`

For a "parameter-linear" edge, meaning an elementwise nonlinearity applied to a linear map `v_hat_i = f(theta_i v_i)`, the derivatives are `d v_hat_i / d v_i = f'(theta_i v_i) theta_i^T` and `d v_hat_i / d theta_i = f'(theta_i v_i) v_i^T`, and both updates become local and Hebbian. Same F, same f'. This paper explicitly builds predictive-coding CNNs, RNNs, and LSTMs from this rule and shows the PC-CNN matches backprop on SVHN, CIFAR-10, and CIFAR-100 (Section 4.2). So predictive coding on convolutional and branching graphs is established, not speculative.

Two practical points from this paper that matter for us.
1. The fixed-prediction assumption (from Whittington and Bogacz 2017). During inference you hold the feedforward predictions fixed and only relax the activities, and you initialize each activity at its feedforward value. This stabilizes inference and is what makes the local rule match backprop.
2. A node with several children sums the contributions over ALL children. It does not average them.

### 1c. Where this all comes from

The origin is Rao and Ballard 1999 (hierarchical predictive coding of visual cortex). The explicit, implementable equations and the free-energy derivation are in Bogacz 2017 (a tutorial, with the exact update equations) and Whittington and Bogacz 2017 (PCN with local Hebbian plasticity). The two arXiv papers above build directly on those. Bogacz 2017 and Whittington and Bogacz 2017 are journal papers, not on arXiv, but the arXiv papers reproduce their equations, which is what I quoted.

---

## 2. The gap - exactly where the repo departs from the canonical rules

I read `dense_pcn_layer.py`, `conv_pcn_layer.py`, and `encoder_encoder_pcn.py`. There are three independent departures, any one of which breaks the single-energy property. Together they guarantee no shared F exists.

### 2a. The activation derivative is the wrong function (the cleanest bug)
The forward pass uses ReLU (`__call__` applies `tf.nn.relu`). But the error term for ReLU layers multiplies by `self.d_gelu(...)`, the derivative of GELU, not the derivative of ReLU.
- `dense_pcn_layer.py` `pred_loss_d_input`, ReLU branch, multiplies the error by `self.d_gelu(self.net_in(x))`.
- `dense_pcn_layer.py` `update_wts`, ReLU branch, uses `self.d_gelu(self.net_in(...))` in the `d_state` term.
- `conv_pcn_layer.py` `pred_loss_d_input` and `update_wts` do the same with `self.d_gelu(...)`.

Canonically this factor must be `f'` of the forward activation. For ReLU that is `1[net_in > 0]`, i.e. `tf.cast(self.net_in(x) > 0, tf.float32)`. Using GELU's derivative for a ReLU forward means the inference and learning updates are gradients of two different (and neither correct) functions. We already tested fixing just this (correcting the ReLU derivative) and the energy still climbed, which tells us this is necessary but not sufficient. The other two departures also have to go.

### 2b. The state update is not minus the gradient of any single F
In the canonical inference rule (1a, 1b) the state update of a node is `-eps_i + sum over children (eps_child . f'(net_child)) W_child^T`. The repo's `DensePCNLayer.update_state` instead does two separate blocks.
- A next-layers block that accumulates `pred_loss_d_input` and `(state - pred_state)` over the next layers and divides by `2 * num_next_layers`.
- A prev-layer block that adds `-(1 + is_clamped)(relu(prev.predict_next()) - relu(self.predict_prev())) @ W` plus a `(predict_next - self(prev.predict_next()))` term, divided by 2.

This departs in several ways at once. It AVERAGES over children (the `/ num_next_layers`) where the gradient SUMS. It applies an extra `/ 2`. It adds a separate prev-layer term that is not part of `-dF/dx_i` for the F that the next-layers block implies. And it mixes ReLU and the GELU derivative across the two blocks. There is no scalar F whose negative gradient is this expression. The conv layer's `update_state` has the same shape and the same problem.

### 2c. The weight update is rescaled, so it is not minus the gradient of F either
`DensePCNLayer.update_wts` (and the conv version) compute a gradient-like quantity and then apply the LARS trust-ratio step we added earlier, `g = (d_state + d_pred)/denom; trust = norm(W)/(norm(g)+1e-6); W -= lr * trust * g`. LARS renormalizes the per-layer step magnitude. Even if `g` were exactly `dF/dW`, multiplying by `trust` makes the actual update a rescaled direction, not `-lr dF/dW`. That was a deliberate symptom-tamer for the weight explosion, but it means the weight phase is no longer descending F. It has to be removed when F is fixed (the correct F-based weight update does not need it).

### Conclusion of section 2
`update_state` and `update_wts` cannot both be `-d/d(.)` of one F as written, because of 2a (wrong f'), 2b (no F has that state gradient), and 2c (the weight step is rescaled). This is consistent with the empirical four-orders-of-magnitude energy climb. It is a structural mismatch, not a tuning problem.

---

## 3. Derivation for THIS architecture

Define one energy, then read off every update as its negative gradient. I use the feedforward-prediction convention of 1a and 1b (each node is predicted from its parents) because it has the proven energy-descent guarantee.

### 3a. The single energy F
- Clamped nodes. The image input `x_img` is clamped to the (normalized) image and the text input `x_txt` is clamped to the caption. They are never updated, they only serve as parents.
- Latent nodes. Every conv activity, every transformer activity, and every `Dense(100)` and `Dense(big)` state is a free latent `v_i`.
- Predictions. Each latent is predicted from its parent(s). For a dense node, `v_hat = relu(W v_parent + b)`. For a conv node, `v_hat = relu(conv2d(v_parent, W))`.
- Shared-latent nodes (the cross-modal coupling). A shared node `v_s` (the ones tied by `share_state_layer`, five of them) has TWO parents, one in the image branch and one in the text branch. It therefore contributes TWO error terms to F, one per incoming prediction.

So
`F = sum over single-parent latents i of ||v_i - relu(W_i v_parent(i) + b_i)||^2  +  sum over the 5 shared nodes s of ( ||v_s - relu(W_s^img a_img + b_s^img)||^2 + ||v_s - relu(W_s^txt a_txt + b_s^txt)||^2 )`

where `a_img` and `a_txt` are the image-side and text-side parent activities of the shared node. This single F is exactly the quantity we have been measuring as the "prediction-error energy." The whole point is that EVERY update below is the negative gradient of THIS F, with no averaging, no extra terms, and a consistent ReLU derivative.

### 3b. Dense layer, concrete updates
Let `net = W v_parent + b`, `v_hat = relu(net)`, `eps = v - v_hat`, and `r = 1[net > 0]` (the ReLU derivative, elementwise). Then

- State update of this node `v` (it is a "child" of its parent, and a "parent" of its own children c):
  `v <- v - eta_x ( eps - sum over children c of (eps_c . r_c) W_c^T )`
  The first term `eps` is this node's own prediction error. The sum is the top-down correction from the children, each child's error times that child's ReLU derivative, projected back through the child's weights. For a shared node, `eps` is replaced by the sum of its per-modality errors `eps_img + eps_txt`.
- Weight update:
  `W <- W + eta_W ( (eps . r) v_parent^T )`
- Bias update:
  `b <- b + eta_b ( eps . r )   (averaged over the batch)`

`r` is the SAME `1[net>0]` in the state, weight, and bias updates. That is the property the repo lacks.

### 3c. Conv layer, concrete updates
Let `net = conv2d(v_parent, W)` (VALID), `v_hat = relu(net)`, `eps = v - v_hat`, `r = 1[net > 0]`.
- State update of the parent (the lower conv map), the child-correction term is a transposed convolution of the gated error:
  `contribution to v_parent  =  conv2d_transpose( eps . r , W )`   shaped back to `v_parent`
  combined with the parent's own `eps` exactly as in 3b.
- Weight update (this is precisely the PC-CNN rule built in 2006.04182 Section 4.2):
  `W <- W + eta_W * Conv2DBackpropFilter( input = v_parent, out_backprop = eps . r )`
- Conv has no bias here, so nothing to do for `b`.

Again `r = 1[net>0]` is shared between the state-correction and the weight update.

### 3d. The transformer branch is the hard part (flagged, see Section 5)
The dense and conv derivations above are textbook. The transformer blocks are not parameter-linear (they contain softmax attention, the add-and-normalize layers, and the sequence-resizing transposes), so the clean local Hebbian rule does not apply to them directly. The implementable options are (i) treat each whole transformer block as one differentiable edge `v_hat = block(v_parent)` and obtain its contribution to `-dF/dv` and `-dF/dW` by automatic differentiation through the block (the DAG framework of 2006.04182 allows any differentiable edge via `d v_hat_j / d v_i`), or (ii) replace the transformer text branch with a parameter-linear stack for a first working version. Either way the honest position is that the text branch is where the established local rule stops applying.

### 3e. The cleanest implementation strategy
Rather than hand-deriving and hand-coding each gradient (which is exactly how the current bugs crept in), define F once as a single scalar function and obtain every `-dF/dv` and `-dF/dW` by automatic differentiation. This is precisely how the maintained libraries work (Section 6). In this repo, since the weights are deliberately non-trainable `tf.Variable`s updated by hand, the sound approach is to compute F with `tf.GradientTape` over the states and the weights and apply `state -= eta_x * tape.gradient(F, state)` and `W -= eta_W * tape.gradient(F, W)`. Autodiff guarantees the two updates are gradients of the same F and that `f'` is always consistent with `f`, which removes departures 2a, 2b, and 2c in one stroke and handles the transformer (3d option i) for free.

---

## 4. Staged plan (do not start on the 7.7B model)

### Stage 0 - prove the derivation on a tiny model first
Build a 2 or 3 layer fully-connected PCN on MNIST or Fashion-MNIST, separate from this repo, implementing 3b with a consistent ReLU (or tanh) derivative. Confirm the single energy F DECREASES across weight updates and that test accuracy rises. This is the proof the rules are right. Do it before touching the big model. The reference equations are 1a (2207.12316 eqs 1 and 2) and the tutorial Bogacz 2017. The fastest version is to run the PCX MNIST example (Section 6), confirm its energy drops, and read how it is structured. Success criterion is energy down, not "no NaNs."

### Stage 1 - port the corrected rules into this repo
Replace `update_state`, `update_wts`, `update_b` in `dense_pcn_layer.py` and `conv_pcn_layer.py` with the 3b and 3c rules (or, better, the autodiff-of-F approach in 3e), remove the LARS rescaling (2c), and use `1[net>0]` everywhere instead of `d_gelu` (2a). Keep the inputs clamped and use the fixed-prediction initialization (initialize each state at its feedforward value, from Whittington and Bogacz 2017 via 2006.04182). Test on a single image-caption pair with normalized inputs and confirm F decreases over relaxation-plus-weight steps on the conv and dense parts. Handle the transformer branch by 3d (autodiff the block, or stub it out for this stage).

### Stage 2 - only after the energy descends, fix the secondary issues
These were documented in REPO_ANALYSIS.md and do not block correctness, only quality and cost.
- Replace the `flatten(full-res conv map) -> Dense(100)` heads with global average pooling or a 1x1 conv before the bottleneck. This is what makes the model 7.7B params for no benefit.
- Use the standard attention scale `1/sqrt(d_k)` instead of `1/d_k`.
- Normalize the image pixels (they are raw 0 to 255 from cv2).
- Reconsider character-level captions versus word or subword tokens.

A relevant aid for Stages 0 and 1 is Salvatori et al. 2022 (arXiv 2212.00720), a stable, fast, automatic PCN learning algorithm, which addresses inference scheduling and stability, and Qi et al. 2025 (arXiv 2506.23800) on training deeper PCNs.

---

## 5. Honest assessment - where the solid theory ends and the guessing starts

Solid and textbook.
- Predictive coding on MLPs. Fully worked out (Bogacz 2017, Whittington and Bogacz 2017, arXiv 2207.12316). The dense-layer derivation in 3b is not research, it is implementation.
- Predictive coding on CNNs. Established. arXiv 2006.04182 Section 4.2 builds a PC-CNN that matches backprop on CIFAR, and Salvatori et al. 2021 (arXiv 2103.03725) show PC can do exact backprop on convolutional and recurrent networks. The conv derivation in 3c is standard.
- Predictive coding on arbitrary DAGs, including a node with multiple parents (the shared latent). Established. arXiv 2006.04182 (arbitrary computation graphs) and Salvatori et al. 2022 (arXiv 2201.13180, learning on arbitrary graph topologies). The shared-latent term in 3a is a legitimate instance of graph predictive coding.

Genuinely research-open or extrapolation, flagged honestly.
- Predictive coding through transformer attention. My arXiv searches for predictive coding combined with transformer or attention returned no core PC-attention theory paper, only tangential hits. Attention, softmax, and the add-normalize layers are not parameter-linear, so the clean local Hebbian PC rule does not apply to them. The DAG framework can still handle them by autodiff of the energy through the block, but there is no established local biologically-plausible PC rule for attention. This is the part of the model with the least theoretical support, and any local derivation here is extrapolation.
- Cross-modal, shared-latent multimodal predictive coding for image and text specifically. The mechanism (a shared node with two parents) is covered by graph PC, and bidirectional predictive coding is actively studied (Oliviers, Tang, Bogacz 2025, arXiv 2505.23415, which I located but did not read in full), but a benchmarked image-text PC model of this exact shape is not something I found in the literature. The idea is reasonable, the specific instantiation is novel and unproven.

Bottom line of the assessment. Re-deriving a clean F and the dense and conv updates is textbook and will fix the divergence for those parts. The transformer text branch is where you leave solid ground. The pragmatic move is the autodiff-of-F strategy (3e), which is theoretically clean for the whole graph and is exactly what the maintained libraries do.

---

## 6. Cross-check - how the maintained libraries implement the updates

The pattern in every maintained library is the same and is the lesson for this repo. Define ONE energy and get both the state and weight updates as gradients of that same energy, rather than hand-coding each gradient.

- PCX, JAX, github.com/liukidar/pcx. This is the closest reference. It represents latent states as "vode" nodes, defines the predictive-coding energy, and computes energy gradients with JAX autodiff (`jax.grad`), so the inference (state) update and the learning (weight) update are by construction gradients of the same energy. The paper experiments, including MNIST and CIFAR, are in the repo's `examples/` folder. Reading one example end to end shows the clean structure to copy. Paper, Pinchetti et al. 2024 (arXiv 2407.01163).
- jpc, JAX, github.com/thebuckleylab/jpc (Buckley lab). "Flexible Inference for Predictive Coding Networks in JAX." A smaller, very readable alternative with the same autodiff-of-energy design. Good for seeing the inference loop and the energy in a compact form.
- ngc-learn and ngc-museum, JAX, github.com/NACLab/ngc-learn and github.com/NACLab/ngc-museum. Component-based rather than autodiff. The most relevant runnable examples are the discriminative predictive coding model (Whittington and Bogacz 2017), documented at `ngc-learn` docs `museum/pcn_discrim`, and the generative GNCN-t1 model (Rao and Ballard 1999) in ngc-museum. These show the explicit error-unit and update-rule components if you prefer to see the equations written out rather than autodiffed.

Recommendation for the cross-check. Open the PCX `examples/` MNIST script first. It is the most direct template for "one energy, autodiff both updates," which is the fix. Use the ngc-museum GNCN-t1 model if you want the equations spelled out as components.

---

## Sources

arXiv papers, pulled via the arXiv API and (for the first two) read as PDFs for the actual equations.
- [2207.12316](https://arxiv.org/abs/2207.12316) - Millidge, Song, Salvatori, Lukasiewicz, A Theoretical Framework for Inference and Learning in Predictive Coding Networks (chain equations, energy-descent proof). PDF https://arxiv.org/pdf/2207.12316
- [2006.04182](https://arxiv.org/abs/2006.04182) - Millidge, Tschantz, Buckley, Predictive Coding Approximates Backprop along Arbitrary Computation Graphs (DAG equations, PC-CNN and PC-RNN). PDF https://arxiv.org/pdf/2006.04182 . Code https://github.com/BerenMillidge/PredictiveCodingBackprop
- [2107.12979](https://arxiv.org/abs/2107.12979) - Millidge, Seth, Buckley, Predictive Coding: a Theoretical and Experimental Review
- [2201.13180](https://arxiv.org/abs/2201.13180) - Salvatori, Pinchetti, Millidge, Song et al., Learning on Arbitrary Graph Topologies via Predictive Coding
- [2103.03725](https://arxiv.org/abs/2103.03725) - Salvatori, Song, Lukasiewicz, Bogacz, Predictive Coding Can Do Exact Backpropagation on Convolutional and Recurrent Neural Networks
- [2212.00720](https://arxiv.org/abs/2212.00720) - Salvatori, Song, Yordanov, Millidge et al., A Stable, Fast, and Fully Automatic Learning Algorithm for Predictive Coding Networks
- [2505.23415](https://arxiv.org/abs/2505.23415) - Oliviers, Tang, Bogacz, Bidirectional Predictive Coding (located, not read in full)
- [2506.23800](https://arxiv.org/abs/2506.23800) - Qi, Forasassi, Lukasiewicz, Salvatori, Towards the Training of Deeper Predictive Coding Neural Networks
- [2407.01163](https://arxiv.org/abs/2407.01163) - Pinchetti et al., Benchmarking Predictive Coding Networks - Made Simple (the PCX paper)

Journal papers (not on arXiv) that are the origin of the explicit equations.
- Rao and Ballard, 1999, Predictive coding in the visual cortex, Nature Neuroscience.
- Bogacz, 2017, A tutorial on the free-energy framework for modelling perception and learning, Journal of Mathematical Psychology (explicit PCN update equations).
- Whittington and Bogacz, 2017, An approximation of error backpropagation in a predictive coding network with local Hebbian plasticity, Neural Computation (the fixed-prediction assumption).
- Vaswani et al., 2017, Attention Is All You Need (standard 1/sqrt(d_k) attention scale).

Maintained libraries and the specific files to read.
- [PCX (pcx)](https://github.com/liukidar/pcx) - read the MNIST script in `examples/`, energy via `jax.grad`, states are vode nodes.
- [jpc](https://github.com/thebuckleylab/jpc) - compact JAX reference, autodiff of the energy.
- [ngc-learn](https://github.com/NACLab/ngc-learn) and [ngc-museum](https://github.com/NACLab/ngc-museum) - discriminative PCN at docs `museum/pcn_discrim`, generative GNCN-t1 (Rao and Ballard) in ngc-museum.
- [predify](https://github.com/miladmozafari/predify) - PyTorch, retrofits PC dynamics onto an existing CNN (for the vision branch).

---

# Deeper arXiv Review (round 2)

A second, wider arXiv sweep (sorted by recency to catch 2024-2026 work), seeded with two papers the user flagged. PDFs of the two seeds were read in full. This section refines, and in places changes, the plan above.

## 1. The ReLU / d_gelu verdict (from arXiv 2505.22074, read in full)

Paper. "The Resurrection of the ReLU" (arXiv 2505.22074), Horuz, Kasenbacher, Higuchi, Kairat, Stoltz, Pesl, Moser, Linse, Martinetz, Otte, University of Lübeck and collaborators. Note this is a vision / activation-function group, NOT a predictive-coding group.

What it actually says. It introduces SUGAR (Surrogate Gradient for ReLU). You keep the standard ReLU in the FORWARD pass, but in the BACKWARD pass you replace ReLU's hard derivative with a smooth surrogate (GELU-style, SiLU, ELU, or their new B-SiLU and NeLU). It is implemented with Forward Gradient Injection using the stop-gradient operator, so the forward output is exactly ReLU(x) while the gradient flows through the smooth function. Their equations.
- Indirect FGI, `y = f(x) - sg(f(x)) + sg(ReLU(x))` (their eq 4), where sg is stop-gradient.
- Direct, `m = x . sg(f_tilde(x)); y = m - sg(m) + sg(ReLU(x))` (their eqs 5-6), where `f_tilde` is the chosen surrogate derivative.
Empirically SUGAR fixes the "dying ReLU" problem and acts as a regularizer, improving generalization on VGG-16, ResNet-18, Conv2NeXt, and Swin Transformer.

So is the repo's d_gelu-on-ReLU intentional or a bug. Both, and the distinction is the whole point.
- It echoes a REAL, named technique. Using a smooth GELU-style derivative for a ReLU forward is exactly surrogate-gradient learning, a well-established idea from spiking nets (Neftci et al. 2019, Zenke and Vogels 2021) that SUGAR ports to ReLU. So the original author's instinct was not arbitrary.
- But in THIS repo it functions as a bug. Three reasons. (a) This repo does PREDICTIVE CODING, not backprop. In PC the activation derivative inside update_state and update_wts is part of the gradient of the energy F, and the energy-descent guarantees (Bogacz 2017, arXiv 2207.12316, and bPC below) require f' to be the TRUE derivative of the forward f so that both the state and weight updates are gradients of ONE F. SUGAR is a backprop regularizer with empirical benefits and NO analogous guarantee in the PC energy framework, so substituting d_gelu breaks the single-energy property, which is the divergence we already proved. (b) SUGAR is a carefully CONTROLLED surrogate (stop-gradient, FGI) where the forward stays exactly ReLU and only the backward is smoothed; the repo instead drops d_gelu in as an ad-hoc factor in hand-derived equations, which is not the SUGAR construction. (c) The repo also has the other structural departures (averaging over children, LARS), so the surrogate framing does not rescue it regardless.

Explicit recommendation for Stage 1. Use the TRUE ReLU derivative `1[net>0]` so the energy provably descends, and make the activation/derivative CONFIGURABLE (default = the consistent true derivative). Reasons. The consistent derivative is required for the energy-descent proof that Stage 0/1 verify. Best of all, keep Stage 0's autodiff-of-F approach, which makes f' automatically consistent with f, so the d_gelu bug literally cannot occur. Surrogate gradients in PC (SUGAR-style) are then a legitimate RESEARCH ABLATION to try ONLY after the consistent baseline trains and the energy descends, knowing it forfeits the clean energy guarantee and must be judged empirically (monitor the energy). Do not keep d_gelu as the default.

## 2. Bidirectional predictive coding (arXiv 2505.23415, read in full) - this changes the plan

Paper. "Bidirectional predictive coding" (bPC), Gaspard Oliviers, Mufeng Tang, Rafal Bogacz, Oxford MRC Brain Network Dynamics Unit (the same group as the framework paper 2207.12316). This is the single most relevant paper for this model.

What it contributes. It unifies generative (top-down) and discriminative (bottom-up) predictive coding into ONE energy with both prediction directions, so a single network does both classification and generation, and it explicitly handles bimodal / multimodal architectures with a shared latent. Its single energy (their eq 3) is
`E = sum_{l=1..L-1} (alpha_gen/2) ||x_l - W_{l+1} f(x_{l+1})||^2  +  sum_{l=2..L} (alpha_disc/2) ||x_l - V_{l-1} f(x_{l-1})||^2`
with W = top-down (generative) weights, V = bottom-up (discriminative) weights, and alpha_gen, alpha_disc scalar precision weights (kept constant, with alpha_disc > alpha_gen because top-down errors are larger). Inference and learning are both gradients of this same E (their eqs 4 and 6), and both contain the same f' (consistency again), with a feedforward bottom-up sweep used to initialize the states.

How it changes PCN_FIX_PLAN.md. The repo's bidirectional, shared-latent, multimodal design is essentially a (broken) instance of what bPC formalizes correctly. Concretely.
- Adopt the bPC energy (their eq 3) as the single F for the multimodal shared-latent part, rather than the one-directional energy I wrote in Section 3a. The shared latent is exactly bPC's shared top layer, which the paper shows shapes a well-behaved energy landscape (their Section 4.4) and prevents the overconfident or biased landscapes that pure discriminative or pure generative PC produce.
- Precision weighting. Give the image-branch and text-branch error terms their own scalar weights (the alphas), because their error magnitudes differ. This is a concrete stability and balance lever the current model lacks.
- Consider separate up and down weights (W and V) rather than tying them via W and W-transpose as the repo effectively does. bPC argues separate weights work better than the shared-weight bidirectional model of Qi et al. (2023).
- Multimodal is no longer "extrapolation." bPC demonstrates a working bimodal model, so the boundary in Section 5 of the plan softens (see point 4 below).

## 3. Other round-2 techniques that improve the odds of this deep multimodal PCN training

- Better initialization. "Faster Predictive Coding Networks via Better Initialization" (arXiv 2601.20895, Pinchetti, Frieder, Lukasiewicz, Salvatori). Initialization schemes that speed and stabilize PC inference. Apply at Stage 1 to set the starting states well.
- Scaling depth. "muPC: Scaling Predictive Coding to 100+ Layer Networks" (arXiv 2505.13124, Innocenti, Achour, Buckley) uses a maximal-update parameterization to train very deep PCNs, and "Towards the Training of Deeper Predictive Coding Neural Networks" (arXiv 2506.23800, Qi, Forasassi, Lukasiewicz, Salvatori) tackles the well-documented degradation beyond 5 to 7 layers. This model is very deep, so depth-stable parameterization is directly relevant and warns that naive deep PC degrades.
- Stability and step-size bounds. "Tight Stability, Convergence, and Robustness Bounds for Predictive Coding Networks" (arXiv 2410.04708, Mali, Salvatori, Ororbia) gives rigorous conditions on the inference and learning step sizes. Use to pick beta and alpha rather than guessing.
- Incremental PC (iPC). "A Stable, Fast, and Fully Automatic Learning Algorithm for Predictive Coding Networks" (arXiv 2212.00720, Salvatori, Song, Yordanov, Millidge, Xu) changes the temporal scheduling so weights update during inference (incrementally) rather than only at equilibrium, which is faster and more stable. A good option once the baseline works.
- Constrained-optimization view. "Augmented Lagrangian Predictive Coding" (arXiv 2605.31022, Seely, Gould) reframes the weight update to better align with the energy minimization, improving stability. Newer, worth watching.
- Classification-plus-reconstruction tension. "Classification and Reconstruction Processes in Deep Predictive Coding Networks: Antagonists or Allies?" (arXiv 2401.09237, Rathjens, Wiskott) studies exactly the situation in this repo, where shared intermediate layers carry both a classification-like and a reconstruction objective, and questions whether they help or fight. Relevant caution for the shared-latent heads.
- Accuracy-deterioration pathology. "Preventing Deterioration of Classification Accuracy in Predictive Coding Networks" (arXiv 2208.07114, Kinghorn, Millidge, Buckley) documents inference accuracy that peaks then declines, with mitigations. Watch for this when evaluating.
- Library. The PCX paper "Benchmarking Predictive Coding Networks - Made Simple" (arXiv 2407.01163) remains the reference implementation for sound, autodiff-of-energy PC with benchmarks.

## 4. Updated honest boundary - does any NEW work close the "PC through attention" gap

No. The round-2 sweep for predictive coding combined with transformer or attention again returned no paper that derives a local predictive-coding learning rule through softmax attention. The nearest hits use "predictive" loosely (a spiking transformer with "active predictive filtering", arXiv 2605.08270; visual-token reduction "via predictive coding", arXiv 2604.00886; slot attention) and none give a PC energy or local rule for attention. So the transformer text branch remains the genuinely research-open part, and the autodiff-of-F strategy (differentiate the energy through the attention block) is still the only sound path there.

What DID move. Multimodal and bidirectional PC is no longer "thin extrapolation" as Section 5 of the plan said. bPC (arXiv 2505.23415) is a 2025 working formulation with a shared-latent bimodal model, so the multimodal shared-latent core of this design now has direct, recent theoretical support. The remaining frontier is specifically PC through attention, not bidirectionality or shared latents.

## Sources (round 2)

Read in full (PDFs).
- [2505.22074](https://arxiv.org/abs/2505.22074) - The Resurrection of the ReLU (SUGAR surrogate gradient). PDF https://arxiv.org/pdf/2505.22074
- [2505.23415](https://arxiv.org/abs/2505.23415) - Bidirectional Predictive Coding (Oliviers, Tang, Bogacz). PDF https://arxiv.org/pdf/2505.23415

Used from abstracts / equations as cited.
- [2505.13124](https://arxiv.org/abs/2505.13124) - muPC: Scaling Predictive Coding to 100+ Layer Networks
- [2506.23800](https://arxiv.org/abs/2506.23800) - Towards the Training of Deeper Predictive Coding Neural Networks
- [2601.20895](https://arxiv.org/abs/2601.20895) - Faster Predictive Coding Networks via Better Initialization
- [2410.04708](https://arxiv.org/abs/2410.04708) - Tight Stability, Convergence, and Robustness Bounds for PCNs
- [2212.00720](https://arxiv.org/abs/2212.00720) - A Stable, Fast, and Fully Automatic Learning Algorithm for PCNs (iPC)
- [2605.31022](https://arxiv.org/abs/2605.31022) - Augmented Lagrangian Predictive Coding
- [2401.09237](https://arxiv.org/abs/2401.09237) - Classification and Reconstruction Processes in Deep PCNs: Antagonists or Allies?
- [2208.07114](https://arxiv.org/abs/2208.07114) - Preventing Deterioration of Classification Accuracy in PCNs
- [2602.07697](https://arxiv.org/abs/2602.07697) - On the Infinite Width and Depth Limits of PCNs
- [2411.02001](https://arxiv.org/abs/2411.02001) - Local Loss Optimization in the Infinite Width: Stable Parameterization of PCNs
- [2407.01163](https://arxiv.org/abs/2407.01163) - Benchmarking Predictive Coding Networks - Made Simple (PCX)
