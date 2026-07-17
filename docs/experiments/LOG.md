# Experiment log

Newest on top. One entry per run or outcome. Never edit past entries.

## Entry format
```
### YYYY-MM-DD short title
- config or command, <what was run>
- result, <the number or observation>
- takeaway, <one line>
```

---

### 2026-07-16 the root-cause chain, flatten bug (original-commit provenance), edge healing, and the NLMS cure
- config or command, the top-to-bottom analysis the lead demanded, executed as a chain, content-trace probe -> flatten pred_loss_d_input bug fix (present since the ORIGINAL full-model commit 754106d of 2026-06-07, invisible at recon rates, the gate was banked with it, decode-rate protocols amplified it 2500x) -> per-edge differential-gain probe -> distillation cells (wd0, trust-cap, iso-on-td at 50 and 200 epochs) -> NLMS conditioning
- result, the spine severing was the flatten bug (image-level content transport 0.046 -> 14-16 after the fix), the remaining defect is five expanding branch edges whose copy-init starts at 30-60x round-trip gain, iso-on-td at 50 epochs healed them 5-10x (rms-boost five-branch decode reached mse 0.0646, under the chance bar, near the 0.0523 single-branch benchmark with edges only partially healed) but 200 epochs REVERSED (11630x, negative cosines), the iso flow and the trust-scaled step fight non-monotonically. The structural cure is NLMS conditioning for the td steps (normalized delta rule, lr*g over input power, local and stable by construction), replacing the trust ratio that provably explodes on tiny-target edges
- takeaway, the generation blocker decomposed into three findable, fixable defects (a five-week-old one-line bug, a pathological copy-initialization, and a mis-matched step conditioner), none of which any amount of objective-level iteration could have fixed. The NLMS distillation is running; if the gains land near 1 the distill-then-join sequence follows, ordinary joint local training warm-started from healthy weights, which retires the phased-training caveat

---

### 2026-07-15 the decode-quality line CONVERGES, every lever past the 50-epoch optimum regresses
- config or command, the full tuning campaign around the untied distilled decode, epoch scaling (150ep), scheduled sampling (tdcasc), the affine per-edge inverse (c_td), the multi-branch boost, and sequential layerwise distillation (tdseq, 5 frozen top-down stages, the textbook composition cure, job 9085, timed out at ep43 with the bottom stage ~30% trained), each retested with the rms-boost readout
- result, the BEST remains ckpt_untied5 (plain 50-epoch teacher-forced distillation) at mse 0.0523, contrast 0.71, coarse per-image luminance (~24 percent of variance, clearly above chance, far from recognizable). EVERY subsequent lever regressed, 150ep 0.0881, scheduled sampling ~black, affine 0.1040, multi-branch 0.149, sequential curriculum 0.1117. The recurring root is the COMPOSITION CURSE, per-edge local objectives do not compose into a good end-to-end decode, and even the composition-aware curriculum could not break it at this cost tier
- takeaway, the principled moves at reasonable cost are EXHAUSTED. What this architecture's local-PC decode achieves on the 2k overfit is coarse per-image luminance, the first above-chance generation of the project and the ceiling of this line. The remaining ideas are multi-day architecture additions (nonlinear per-edge decoders, trained branch fusion) carrying the same composition risk. The scientific yield stands, tying proven to be the blocker via eight mechanisms, the untied amendment validated with the first content ever through the cascade, the readout solved (rms matching), and the local optimum precisely characterized

---

### 2026-07-13 the untied arc, first content through the cascade, then the scheduled-sampling regression
- config or command, the constraint amendment (untied top-down weights wts_td per image-path edge, local d_pred training, opt-in --untied byte-identical off). Full untied training destabilized after ep1 at three stabilizer settings (though it FIT FASTER than tied, floor 0.0034 at ep1). Phase-2 decode DISTILLATION (--train-mode tdonly, encoder frozen, teacher-forced per-edge td steps, stable by construction) on the isotropic encoder, then scheduled-sampling (tdcasc, 50/50 cascade-input batches)
- result, the teacher-forced distillation (ckpt_untied3) produced the FIRST CONTENT EVER through the cascade, the boost readout shows clear per-image coarse luminance structure tracking each true image (the all-white shower column is the tell), not recognizable scenes but the first non-template decode of the campaign. The scheduled-sampling iteration then REGRESSED, the cascade-input mix taught the edges to shrink on saturated states, the boost readout went black while the free cascade stayed saturated (gain 10 to 3.6). Best artifact remains ckpt_untied3(+_td)
- takeaway, the untied amendment is validated at the mechanism level (tying was the blocker, a dedicated trained decode carries content) and the remaining work is now HYPERPARAMETER iteration (mix ratios, schedules, readout calibration) rather than mechanism discovery. Recorded as the inflection point for the paper's tied-vs-untied ablation

---

### 2026-07-13 the isometry constraint trains free and stabilizes, but generation is still content-blind, the FINAL null
- config or command, scaled semi-orthogonalization flow in update_wts (small-side Gram -> c*I, anisotropy killed, scale left to the recipe; two pre-flight fixes, small-side Gram for the expanding 100->98304 inters and the scaled target after the unit target collapsed their natural scale 66x). From-scratch COCO64_GEN recon 15ep --isometry 1e-3 (job 9054, HEQ), then the three-cell retest (job 9058, pi / plain / boost)
- result, the constraint itself is a WIN, recon reached the SAME floor as unconstrained (0.006) with LOWER bounded states (~80 vs 120-155), i.e. a free stabilizer. But generation is STILL content-blind, the pure-cascade cells saturate exactly as before (mse 0.4496, image-set == text-set), and the boost cell changed amplitude only (image-set now over-drives to blown-out white 3.2x, text-set rose to a brighter 0.64/0.62 template, the best text-set numbers of the campaign but visually the SAME template in every column, and the swaps are still single-code)
- takeaway, isotropic per-edge geometry fixes the GAIN pathology but content still does not survive the composed cascade, the residual killers being exactly the spec's flagged risks, GELU and the strided downsampling are not inverted by ANY transpose. This was the last mechanism-level swing and it is a null. THE GENERATION CAMPAIGN CLOSES, eight objective/constraint mechanisms and three inference routes, every failure pinned. The transferable wins carry to the ladder, weight-norm (stabilizer), the ISOMETRY CONSTRAINT (a free stabilizer at zero floor cost, directly relevant to the ladder's norm-inflation failure mode), corrected monitoring, and the HEQ scheduling route

---

### 2026-07-13 strong-pressure cascade destabilizes instantly, the calibration family is CLOSED
- config or command, --train-mode cascade at recon-equal pressure (gen-lr 1e-3, gen-every 1, 40x the gentle cell), fresh warm-start, job 9047 on an n15 HEQ slice
- result, wrecked in 50 steps (energy 20898, states pinned at 400 by the first print), even faster than the end-to-end contrast's 150. Cancelled. The pressure curve for per-edge calibration is complete, gentle (1e-4 every 4th) = stable but out-anchored ~40 to 1 by the recon interleave and moves nothing; strong (recon-equal) = the calibration and recon steps fight over the same shared weights at equal strength and the off-manifold cascade targets tear the recon equilibrium apart immediately
- takeaway, the same empty-window shape has now appeared for the THIRD training family (end-to-end CHL contrast, expressing-free-phase calibration, per-edge cascade-consistency), gentle is ineffective, strong destabilizes, and the middle never carries. The root is the SHARED WEIGHTS, any objective that pulls them toward top-down duty at effective strength rips them off the bottom-up equilibrium the recon anchor needs. The calibration family is CLOSED. Remaining, the isometry repair (a constraint shaping the geometry DURING recon rather than an objective fighting it, the one idea that does not enter this tug-of-war) or banking

---

### 2026-07-13 cascade-consistency training is STABLE but INEFFECTIVE at gentle pressure
- config or command, --train-mode cascade (per-edge alignment of the top-down cascade to the recon-state targets, d_pred-only local steps, no anti-learn), job 8954 warm-started from ckpt_gen_best, gen-lr 1e-4 gen-every 4, 12ep with weight-norm; retest job 9046 on an n15 HEQ slice (latent_source_diag at the calibrated pi schedule AND the boost baseline)
- result, training SURVIVED all 12 epochs (energy 0.021, states bounded 156, the only calibration approach that has not destabilized). But the retest is a wash, the pi-schedule readout is STILL content-blind saturation (image-set mean 5.57, mse 0.4496, identical to text-set; the cascade gain came down from ~7.7 to ~5.6 and nothing else changed) and the boost readout is the SAME dim template as before training (0.52/0.24, mse 0.1005 vs 0.0985). Likely why, the gentle calibration (1e-4 every 4th step) is out-anchored ~40 to 1 by the interleaved recon steps (1e-3 every step) on the SAME shared weights, and one W is being asked to serve bottom-up prediction and top-down inversion at different operating points
- takeaway, per-edge credit assignment fixes the STABILITY problem (no compounded error, no destabilization) but at this pressure it does not move the decode. The untested cell is STRONG calibration pressure (gen-lr 1e-3, gen-every 1, up to 40x more), cheap now that the HEQ slices schedule instantly; beyond that the isometry repair is the last mechanism-level idea, then banking. INFRA, n15 undrained (admin confirmed, the drain was an n14/n15 typo), each HEQ = H200 NVL MIG 1g.35gb (~31GB), 12 usually idle, use for all COCO64-scale jobs; the whole H200 (for NATIVE-143 and the big ladder rungs) frees when the other-session p3-control ends

---

### 2026-07-13 the contraction swing is EXHAUSTED, calibration destabilizes at every non-degenerate expression level
- config or command, the calibration CHL (image-set latents, expressing free phase via free_pi_bu/free_state_lr) at two expression levels, full (0.25 x 45 steps, job 8946) and gentle (0.05 x 30 steps + gen_lr 1e-4, job 8949), both warm-started from ckpt_gen_best with weight-norm
- result, BOTH destabilize the same way. Full expression wrecked the recon weights in 150 steps (energy 1957, states pinned 400 by ep0), anti-learning at saturated off-manifold states. Gentle expression degraded on an accelerating curve (energy 0.003 -> 0.005 -> 0.012 -> 0.048 -> 0.386 with states 120 -> 190 -> 400 pinned by ep2, a decade every ~150 steps), the same failure ~5x slower. Cancelled both. With the earlier degenerate level (the original latent-AE CHL, ~zero expression, which FLATTENS instead), the three points bracket the whole window, any free-phase expression strong enough to give a non-degenerate negative sample also destabilizes the shared weights faster than it calibrates them
- takeaway, the CONTRACTION SWING IS EXHAUSTED and the generation question closes as a thoroughly characterized negative. Final tally, the inference half (probe grid) showed no schedule extracts identity from the recon weights (rate-starved zero / boost template / adequate-rate content-blind saturation), and the training half destabilizes at every non-degenerate expression level. Across the whole campaign, seven generative objectives, two inference routes, and the calibration contrast at three expression levels, every failure with a pinned mechanism, all opt-in code byte-identical off. RECOMMENDATION, bank the generation chapter and pivot the remaining time to the banked coupling-ladder question

---

### 2026-07-13 precision probe CORRECTS the record, rate starvation + a miscalibrated cascade, not contraction
- config or command, added per-layer error precisions pi_td/pi_bu to update_state (defaults 1.0 byte-identical, GATE_MATCH nlayers=88) and a --decode-state-lr override to latent_source_diag; read-only probe grid on ckpt_gen_best (jobs 8940/8941/8942), cells = {rate-starved 1e-4, adequate 0.05/0.2} x {pi_bu 0.0/0.3/1.0}, plus the earlier gamma-boost cell
- result, TWO corrections and a final verdict on these weights. (1) The "contractive pathway decodes to zero" was RATE STARVATION, the diagnostics built the model at state_lr=1e-4 so 150 relax steps move states ~1.5 percent of the error, states never left the zero init (training never noticed because pass_through initializes at the forward values). (2) At an adequate rate the cascade EXPRESSES but is MISCALIBRATED, every schedule (pi_bu 0/0.3/1.0) lands on a content-blind saturated output (means 3.4-8.6, ~10-20x too bright, MSE 0.4496 identical everywhere, image-set vs text-set vs swaps differ in the 3rd decimal). So the composed per-edge top-down maps are off-manifold expansive and content-blind, each edge was only fit around bottom-up operating points. VERDICT on ckpt_gen_best, no inference schedule can extract identity from the trained top-down cascade
- takeaway, this also REINTERPRETS the latent-AE CHL failure, its free phase was rate-starved so the negative sample was degenerate (~zero states) and the contrast had nothing to calibrate against. The swing's training half is therefore still live and now precisely aimed, rerun the latent-AE CHL with the free phase at an adequate rate + generative precisions (pi_bu small), so the anti-learn sees the actual miscalibrated cascade and the contrast calibrates the composed inverse around image-set latents (well-posed). Weight-norm's trust ratio guards the large-error steps

---

### 2026-07-12 latent-autoencoder CHL FAILS, the top-down pathway is contractive and the boost is information-destroying
- config or command, --gen-latents image (chl_step phase 0 = recon clamp -> IMAGE-set latents, well-posed target), job 8936, warm-start ckpt_gen_best, 12ep with --weight-norm, stable throughout (energy 0.023, states 156). Retest latent_source_diag on ckpt_chl_ae at gamma 1.0 AND gamma 0.0
- result, (1) gamma=1 decode(img-set) is STILL a template, now DIMMER than before training (0.53/0.26 vs the untrained 1.09/1.31; mse 0.0985 ~ what a flat image scores), no scene identity, and the swaps are unchanged (s0 alone reproduces it, s1-4 inert). The well-posed target produced the SAME low-amplitude flattening as text-CHL. (2) gamma=0 (plain relaxation) decodes to EXACTLY ZERO from every latent source, mean 0.0000 std 0.0000, the image and decode states fully collapse
- takeaway, the root cause is now pinned and it is STRUCTURAL, the top-down pathway of this shared-weight net is CONTRACTIVE (any latent signal dies to zero across the 9-layer decode without a boost), and the only revival route (the gamma=1 replacement boost) is single-chain feedforward that destroys all but the deepest code's information. Training cannot fix this with a local contrast because the free phase itself collapses to zero (a degenerate negative sample). SIX generative objectives (gen, gentle-lr, CHL, HF, EBM, diffusion) plus the well-posed latent-AE CHL have now each failed with a clear mechanism. VERDICT, sharp top-down generation is out of reach for THIS shared-weight bidirectional-PC design at 64px, and the characterization of why is complete. Recommendation to the research lead, BANK this as the paper's generation-side negative-result chapter and pivot the remaining ~2 months to the banked coupling-ladder question

---

### 2026-07-12 latent-source probe REFRAMES the failure, the decode emits a template even from image-set latents
- config or command, tools/latent_source_diag.py on ckpt_gen_best (read-only, L4 job 8934), decode from IMAGE-set latents vs TEXT-set latents under the identical boosted standalone protocol, plus per-scale swaps (text-set with one scale replaced by image-set)
- result, THREE findings. (1) Even from IMAGE-set latents (correct content at every scale) the standalone top-down decode produces only a brightness-modulated LUMINANCE TEMPLATE (bright top, dark bottom, no scene identity; full amplitude 1.09/1.31 but MSE-to-true 0.1354 = same as text-set 0.1361). (2) The per-scale swaps are decisive, swapping the DEEPEST code (s0) alone reproduces the image-set decode EXACTLY while swapping any of scales 1-4 changes NOTHING to 4 decimals, so under the gamma=1 boosted route 4 of 5 latents are completely inert and the 5th only modulates the template brightness. (3) The fine-latent-attenuation hypothesis is REFUTED as the binding constraint, correct fine latents contribute zero through this decode
- takeaway, the top-down decode was NEVER trained to be top-down self-sufficient. Recon always had the image clamped (intermediates carried the image bottom-up, weights only needed the average template), and EVERY generative objective so far (gen/CHL/HF/EBM) conditioned training on TEXT-set latents = the ill-posed conditional-mean setup. The untried, well-posed half is training the decode from IMAGE-set latents (they identify the image, so the conditional mean given them IS the true image, the mean-collapse is removed by construction). NEW plan, (A) latent-autoencoder CHL (chl_step with phase 0 = recon clamp so the latents are image-set, then the same free/anti-learn vs clamped/learn local contrast) to train top-down self-sufficiency, then only if (A) passes, (B) text-to-code alignment (clamp the caption and the latent target, local steps on the text trunk). If (A) fails the decode is fundamentally incapable under shared weights and banking is the honest call

---

### 2026-07-12 Approach B3 (diffusion-in-PC) is a NULL, the denoiser trained into a plain autoencoder
- config or command, diffusion-in-PC: diffusion_step (encode a noised x_t + caption -> latent, local weight step toward the clean x_0) over a 10-level schedule (sigma 0.05..0.8), --train-mode diffusion, warm-started from ckpt_gen_best, job 8902, 10ep, L4, no layer change (noise on data) so gate holds trivially. Retest, reverse-diffusion sampler tools/diffusion_sample.py + direct denoiser test tools/denoise_test.py
- result, training STABLE (energy finite, states bounded, TRAIN_DONE). Reverse-diffusion generation from noise = PURE RGB NOISE (image grid). Fixed a real sampler bug first (it read the reconstruction of x_t with x_t still clamped; corrected to unclamp+decode-relax to read x0_hat) -- still noise. DIRECT DENOISER TEST is conclusive, at every level the denoised MSE-to-true EQUALS the noised MSE EXACTLY (0.0087/0.0610/0.1461), i.e. x0_hat = x_t, the net just RECONSTRUCTS its input and never denoises
- takeaway, B3 is a NULL because the diffusion_step trained a PLAIN AUTOENCODER, not a denoiser. Root cause, bidirectional PC's SHARED WEIGHTS (encoder = decoder^T) default to reconstructing the input, and the clean-recon interleave reinforces it, so the x_0 target never shifts the map off identity. Possible salvage (no recon interleave / predict noise eps / stronger denoising target) but the shared-weight tendency is fundamental. THREE clean nulls now on the mean-collapse (A deterministic HF, B1 Langevin EBM, B3 diffusion), each with a clear mechanistic reason. B3 code stays in the tree (opt-in, byte-identical off)

---

### 2026-07-12 Approach B1 (noisy-relaxation EBM) is a NULL, samples are pure noise
- config or command, PC-native EBM: opt-in Langevin noise in update_state (noise_temp=0 byte-identical, GATE_MATCH nlayers=88) + ebm_step (chl_step with a noisy CD-1-from-data negative phase), --train-mode ebm. Warm-started from ckpt_gen_best, T0 sweep 50 (job 8878) + 300 (job 8879), 10ep, L4. Retest, deterministic darkness_diag AND sampling (tools/ebm_sample.py, annealed noisy relaxation, image grids)
- result, training STABLE (energy 0.013-0.020, states bounded 118-141, weight-norm+noise+clip contained it, TRAIN_DONE both). But GENERATION FAILS both ways. Deterministic darkness_diag on the EBM ckpt is WORSE than plain CHL (0.282 brightness / 0.134 contrast vs 0.40/0.24) -- reading the mode of an EBM is degenerate. SAMPLING (noisy relaxation) gives PURE NOISE, samples mean ~0 (vs true 0.32), std 0.55 (T0=50) / 1.34 (T0=300) vs true 0.22, values -5.7..+6.2 far outside [0,1]; the image grid is RGB static with zero structure at both T0
- takeaway, B1 is a NULL. The CD-1 contrast did not carve sharp low-energy wells, so Langevin sampling has no sweet spot, small noise -> the blurry mode (worse than baseline), large noise -> static. This is the known finicky-EBM failure (deep CD does not mix/sharpen easily). The mean-collapse survives BOTH a deterministic reweighting (Approach A) and a Langevin EBM (B1). Next principled option = the DIFFUSION flavor (B3, multi-noise-level denoising, which explicitly learns noise->data at each level and is the proven cure for blurry generation), the biggest lift. B1 code stays in the tree (opt-in, byte-identical off) as a characterized negative

---

### 2026-07-11 Approach A (HF-weighted energy) does NOT crack the mean-collapse (full gamma sweep flat)
- config or command, high-frequency boost on conv1's bottom pixel error (e -> e + gamma*Laplacian(e), reflect-padded, opt-in --hf-weight, byte-identical off, GATE_MATCH nlayers=88; composes with --weight-norm). Warm-started from ckpt_gen_best, recon+HF (job 8851) and chl+HF gamma sweep 1/2/3 (jobs 8857/8859/8860, --train-mode chl --gen-lr 3e-4 --gen-every 4 12ep, L4). darkness_diag (--weight-norm) on each
- result, recon+HF, NO change (0.382/0.404 = the warm-start baseline; a converged recon has ~0 gradient to reshape and is image-clamped so it never touches the text-driven decode). chl+HF, FLAT across the whole sweep: contrast (std ratio) 0.243 (gamma0/plain CHL) -> 0.251/0.253/0.254 (gamma 1/2/3); brightness 0.40 -> 0.39; latent T/I ratios unchanged. All runs STABLE (energy 0.02, states bounded ~156, no drift to the clip -- HF+weight-norm held better than plain CHL which drifted to 309)
- takeaway, Approach A (a high-pass confined to conv1, the last pixel-mapping) does NOT sharpen text-to-image, even at 3x boost. The blur is set UPSTREAM (the whole decode chain + the shared latent), which a bottom-layer boost cannot reach. This is a FAIR null result over a full sweep and it REINFORCES the mean-collapse diagnosis (a deterministic error-reweighting cannot beat the conditional mean). Escalate to Approach B (PC-native stochastic sampling, user pre-approved), where sampling breaks the mean-seeking by construction. The HF-weight code is opt-in + byte-identical off, so it stays in the tree as a characterized negative

---

### 2026-07-11 weight-norm STABILIZES training but the CHL OBJECTIVE is the ceiling (retest verdict)
- config or command, warm-started CHL+weight-norm (job 8845, ckpt_gen_best -> ckpt_chl_wn, --weight-norm --train-mode chl --gen-lr 3e-4 --gen-every 4, 30ep, L4); darkness_diag (new --weight-norm restore path, tools/darkness_diag.py) on the ep1-best and ep12 checkpoints
- result, STABILITY FIXED, energy stayed bounded (floor 0.008 -> 0.07 at ep15, vs the prior un-normalized CHL exploding to 7-10 by ep19) and it ran PAST the ep13 wall every prior generative attempt hit. A residual STATE drift remains (max_abs_state 103@ep8 -> 336@ep15, ~+30/ep, heading to the 400 clip ~ep18) but energy-bounded, not a blowup. GENERATION NOT IMPROVED and it FLATTENS with training, boosted-gen contrast (std ratio) 0.404@ep1 -> 0.243@ep12; brightness (mean ratio) stuck ~0.38-0.40 (the conditional-mean level) throughout. ep1-best (0.378/0.404) equals the prior CHL best (0.38/0.40), so weight-norm does NOT damp from the start; the CHL training itself flattens the decode over epochs, even as the fine-latent text drive IMPROVES (T/I 0.47/0.58@ep1 -> 0.64/0.82@ep12)
- takeaway, the "norm-inflation is the blocker" hypothesis is REFUTED. Weight-norm fixed the instability (necessary + real; runs past ep13) and that isolated the true ceiling = the GENERATIVE OBJECTIVE. Squared-error PC decode training converges to the blurry conditional mean; more (now-stable) training drives it FLATTER not sharper. Next lever (user-chosen) = a generative objective that rewards high-frequency amplitude / breaks the mean-seeking, within bidirectional PC. The weight-norm code is a KEEPER (stabilizes any longer generative training; byte-identical off, GATE_MATCH held)

---

### 2026-07-11 weight-norm needs the LARS trust ratio KEPT (fresh-init smoke exploded without it)
- config or command, plumbing smoke of --weight-norm on COCO64_GEN (L4), fresh-init recon+weight-norm and chl+weight-norm, per-step energy and max_abs_state
- result, the spec's "drop LARS trust" EXPLODED in one weight step (step1 energy 259 -> step2 5.9e20, states slammed to the 400 clip, NaN by step3). Root cause, weight-norm bounds the ASYMPTOTIC ||w|| but NOT the per-step gradient magnitude; the LARS trust ratio was the per-layer STEP normalization a single lr needs across this model's fan-in (27..20.6M). FIX (commit 7d8f50e), keep the trust ratio in the weight-norm branch, computed on ||wts||=||v|| which the tangential split PRESERVES, so it normalizes the step with no inflation feedback (the radial growth is already diverted into the damped g_mag). After the fix, fresh recon AND chl train STABLY (energy ~0.05, states bounded 8->22 over 4 epochs, TRAIN_DONE); 7 unit tests still pass; off-path untouched so the gate still holds
- takeaway, weight-norm and trust address DIFFERENT failures -- weight-norm kills the asymptotic norm inflation (the slow ep13 creep), trust normalizes the per-step size across the wide fan-in. Both are needed; the spec was wrong to drop trust. The make-or-break is whether the two together hold the warm-started CHL run past ep13

---

### 2026-07-11 weight-norm stabilizer, layers + COCO64 inertness gate (byte-identical when off)
- config or command, implemented the PC-native weight-normalization reparameterization on the conv (c0a4d42) and dense (e0ba22e) layers (w = g_mag * v/||v|| per output unit, weight() accessor both directions, update_wts split into a radial magnitude step + a tangential direction step, LARS dropped for these layers, opt-in --weight-norm); 7 unit tests pass. Gate on L4, tools/rewrite_gate.py --config coco64 --relaxed vs the pre-change banked ref docs/superpowers/gate_ref_coco64.npz
- result, byte-identical when off CONFIRMED: run-to-run GATE_MATCH (current code deterministic on one node) AND ref-vs-cur GATE_MATCH nlayers=88 tol=1e-4. First gate attempt on node n7 showed a spurious GATE_MISMATCH of 1.27-1.32e-4 on 3 of 88 layers = cross-node L4 fp variation (cuDNN), NOT the code; re-running on a node consistent with the ref gave the exact match. Operational lesson, a single-run L4 GATE_MISMATCH at ~1e-4 needs a same-node run-to-run re-check before it counts as a regression (or pin the node / use tol ~2e-4 cross-node). NATIVE-143 gate deferred to a big-GPU window (H200 drained); interim proof = this COCO64 gate + provable inertness (weight() returns self.wts, else-LARS unchanged) + the unit tests
- takeaway, the weight-norm change is inert when off (NATIVE-safe), so the structural fix is in place; next is wiring --weight-norm into train_coco64 (with g_mag persistence) and the make-or-break CHL retrain that must hold training past the ep13 norm-inflation wall

---

### 2026-07-11 CHL destabilizes ~ep13 like every approach: NORM INFLATION is the common blocker
- config or command, CHL retrain (job 8830, warm-start recon, --train-mode chl --gen-lr 3e-4 --gen-every 4, 30 ep, L4); darkness_diag + gen_retest on ckpt_chl (ep29, corrupted) and ckpt_chl_best (stable, early)
- result, CHL held to ~ep10 (energy 0.014, state ~110) then norm-inflated: state hit the 400 clip by ep16-17, energy exploded to 7-10 by ep19-20. Stable best ckpt: standard test_step still a blob; boosted generation has the best CONTRAST yet (std ratio 0.40 vs prior 0.08-0.30) but the same dark brightness (mean ratio 0.38), not recognizable. Monitoring gap: the B-gate grep watched only DIVERGED/RuntimeError, not state-pinned-at-clip, so the Monitor reported a clean TRAIN_DONE on a blown-up run
- takeaway, EVERY generative-training approach (InfoNCE, one-sided bridge, gentle-lr, CHL) destabilizes via NORM INFLATION at ~ep13, capping the achievable generation. The objective is not the blocker; the underlying norm-inflation instability is (the beta-less-LARS / effective-LR-vs-scale thread). The real next lever is the deferred structural stabilizer (muP effective-LR scaling, weight normalization, or a norm penalty in the PC energy) so any generative training can run long enough to sharpen

---

### 2026-07-10 objective diagnostic: darkness is the top-down conditional-mean, not weight decay
- config or command, tools/objective_diag.py A/B on ckpt_gen_best (clean recon), 30 generative steps with weight decay ON (3e-2) vs OFF (0) on the decode layers, measuring decode weight-norm + boosted-generation brightness
- result, decode weight-norm unchanged in both arms (424.5 -> 424.3, ratio 1.000, so weight decay is NOT shrinking the weights); weight-decay-OFF ended darker (gen mean 0.033) than ON (0.087), refuting the shrinkage hypothesis. Even the clean recon decode already generates dark top-down (mean 0.155, 0.40x true) with ZERO generative steps, and generative steps rearrange the weights toward an even darker output
- takeaway, the darkness is the top-down PC decode producing the low-amplitude CONDITIONAL MEAN (the blurry average image consistent with the latent); the current bridge objective (true image clamped at the bottom during the weight step) does not teach a strong latent->image map usable at generation. The lever is a redesign of the generative objective for sharp top-down generation, not tuning / longer training

---

### 2026-07-10 more generative training (gen-every 2) made amplitude WORSE, points at the objective
- config or command, --gen-lr 3e-4 --gen-every 2 warm-started from clean recon (job 8826); drifted (energy peaks 0.010->0.021 ep7-10) and cancelled; darkness retest on the ep8 ckpt (~1000 gen steps, more than gentler-A's 750)
- result, generated brightness 0.186x true (mean 0.071), WORSE than gentler-A's 0.386x; fine-scale text-set latents attenuated (T/I ratios 0.41, 0.57 at scales 4-5 vs ~1.0 in gentler-A). gen-every 2 was a hyperparameter mistake (doubled generative frequency -> drove the drift), so not a fair test of gentle-lr
- takeaway, more generative training via this config shrank the signal rather than sharpening it; the fine-scale attenuation suggests the phase-2 objective may be SHRINKING the decode (hypothesis, weight decay dominating the small-error weight step), a fixable objective flaw worth diagnosing before any longer run

---

### 2026-07-10 darkness diagnostic: the gap is decode-amplitude, not the latent or text path
- config or command, tools/darkness_diag.py on ckpt_gen_warm (gen-trained ep12), compare text-set vs image-set shared-latent scale + generated-image brightness vs true
- result, text-set latents match image-set in scale (norm ratio T/I = 0.89-1.12 across all 5 scales, so the text path sets the latent at the right scale). But the top-down-decoded image is mean 0.148 / std 0.080 vs the true image mean 0.383 / std 0.263 (0.39x brightness, 0.30x contrast, max only 0.357 vs 1.0). The same decode weights produce a bright image in reconstruction (image clamped), so the decode is capable but has not learned to produce full-scale output from the latent top-down
- takeaway, the darkness is a DECODE-TRAINING gap, not a latent-capacity ceiling or text-path failure; the lever is a better/longer/balanced generative-training run (keep recon at the floor, train the decode much longer) to push the output to full brightness/contrast/detail

---

### 2026-07-10 gentler-A generative training: stable, modestly shifts boosted generation, not recognizable
- config or command, gentler A (warm-start from the recon-trained COCO64_GEN best, then --train-mode gen --gen-every 4), job 8820 on L4, then text->image retests on the ep12 checkpoint
- result, trains STABLY for 12 epochs (energy near floor, state bounded ~120-160) then a slow drift crosses the thresholds at ep13 (stopped, kept the ep12 ckpt = ~750 stable generative steps). Standard test_step t2i still blobs (it uses the symmetric relaxation, not the top-down training regime). recon still works (PR 3.0) but degraded (energy 0.023 vs 0.006 floor); image->caption degraded to empty. Top-down-BOOSTED generation on the gen-trained ckpt: PR 6.38 at gamma 1.0 with ~3.4KB content, vs the recon-only model's high PR only as tiny speckle (730B) — so more caption-distinct and content-richer, but visually dark faintly-banded fields, NOT recognizable scenes (the per-caption variation is real but low-amplitude)
- takeaway, the generative-training objective is stable at a gentle schedule and modestly improves the top-down-boosted output, but does not crack recognizable text->image and costs recon+i2t; the remaining obstacle is likely the shared 100-dim latent capacity / the fundamental difficulty of from-scratch PC 64px text->image

---

### 2026-07-10 approach-A generative training destabilizes (recon-vs-generation tension)
- config or command, COCO64_GEN --train-mode gen at gen-every 1, 2k pairs, stable recipe (lr 1e-3 wd 3e-2 clip 400 gelu), job 8819 on the L4
- result, the two-phase generative step interleaves correctly (traces once, clamp hygiene holds) and is stable to ep2 (energy ~0.008, max|state| ~44), then energy climbs 0.008->0.037 and max|state| jumps 44->126 by ep3, accelerating on both; cancelled at step 900
- takeaway, the mechanism is sound (GT.1/GT.2 reviewed clean) but at gen-every 1 the recon-vs-generation pressure destabilizes the shared weights; next lever is a gentler A schedule (recon warm-up then infrequent generative steps) or the soft-nudge Approach B, a gated user decision

---

### 2026-07-09 top-down-boosted generation confirms invertible downsampling reconnected the pathway
- config or command, top-down-boosted test_step generation on COCO64_GEN/ckpt_gen_best, gamma sweep 0/0.5/1/2 (the boost now flows THROUGH the invertible strided-conv downsamplers, unlike the maxpool-blocked earlier attempt), job 8808 on the L4
- result, image PR rose 0.00 -> 1.34 -> 2.32 -> 6.93 (max 8) with gamma; the caption now drives the image per-pixel and distinctly per caption (vs the maxpool model's uniform green tint), but the outputs are transpose-conv checkerboard fields (low gamma) / caption-varying colorful speckle (high gamma), not recognizable scenes
- takeaway, invertible downsampling + a top-down generation schedule solve the structural block AND the drive-balance at inference (caption controls the image, no retraining) -- recognizable CONTENT is the one remaining lever and needs the generative-training objective, since the decode was trained both-clamped and was never top-down-self-sufficient

---

### 2026-07-08 strided-conv downsampling ships but text to image still blobs
- config or command, COCO64_GEN with strided-conv downsampling
- result, downsampling trains stably and ships clean, text to image still blobs under the standard relaxation
- takeaway, invertible downsampling is necessary but not sufficient, drive-balance and non-generative training remain the obstacles

---

### 2026-07-07 PC-native InfoNCE coupling fails to unblock text-to-image
- config or command, PC-native InfoNCE coupling at lambda 0.1, 0.3, and 1.0, validated against the no-InfoNCE baseline
- result, code alignment stayed at chance and text-to-image generation stayed a blob at every lambda
- takeaway, contrasting the paired codes alone does not fix the image-dominated shared latent, the mechanism and follow-ups are recorded for the next attempt

---

### 2026-07-07 fixed the attention pad-mask no-op SP1 review caught
- config or command, moved the mask broadcast in AttentionPCNLayer from the query axis to the key axis
- result, padded caption positions now get excluded for every query, re-gated GATE_MATCH nlayers=143 for NATIVE_7B (still inert there since its mask is all-zeros)
- takeaway, the old broadcast shape let softmax's shift-invariance silently no-op the mask, code review caught what the running tests missed

---

### 2026-07-07 best-checkpoint save reveals coco64 does overfit then destabilize
- config or command, state_clip=150 training run on coco64, checkpointed on every save
- result, energy dropped from 2.17 to 0.13 by epoch 7, then the run destabilized after about epoch 8, and max_to_keep=1 had been overwriting the good weights with the blown-up ones
- takeaway, save the best (lowest-energy) checkpoint separately so the learned model survives a later divergence

---

### 2026-07-07 state-clip added after a state-norm divergence
- config or command, --state-clip arg capping max |state| after relaxation on the dense and conv layers carrying the shared latents
- result, the lower-learning-rate run still diverged by state-norm inflation, the same failure mode seen at 7.7B, the lower rate only delayed it
- takeaway, cap the runaway quantity itself (state norm) rather than trying to out-tune the learning rate around it

---

### 2026-07-07 COCO64_156M validated on the H200
- config or command, instantiate, batch, and generate checks for COCO64_156M on H200 hardware
- result, batches produce finite states, all 5 shared-latent aliases hold, and generation works in both directions
- takeaway, the bidirectional class now has a working 64px config, clearing the way for the coco64 training work

---

### 2026-07-07 COCO64_156M tuned to land near its 156M target
- config or command, added a param counter and a --config flag to the gate tool, then tuned COCO64_156M's widths
- result, the config lands near 156M parameters
- takeaway, the gate tool now reports param counts per config, making future capacity targets a tuning exercise instead of a guess
