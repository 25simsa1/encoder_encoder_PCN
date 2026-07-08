# PC-native InfoNCE coupling — outcome report

Date: 2026-07-07
Plan: `docs/superpowers/plans/2026-07-07-pc-native-infonce-coupling.md`
Spec: `docs/superpowers/specs/2026-07-07-pc-native-infonce-coupling-design.md`

## Goal (recap)

Fix the bidirectional class's text→image failure (image-dominated shared latent →
text→image generates uniform blobs) with a PC-native InfoNCE coupling: contrast the
deepest-scale branch codes (`inter2` image / `inter12` text, 100-dim), inject the
contrastive gradient (computed w.r.t. the codes only) into the code states during
relaxation, weights still learned solely by the existing local LARS. No network
backprop. Opt-in (lambda=0 → NATIVE unchanged, GATE_MATCH intact).

## What shipped (code, all reviewed clean, on master)

- `infonce.py` `infonce_grads(u,v,tau)` — symmetric InfoNCE, gradient w.r.t. codes only
  (tape scoped codes→loss, no network backprop). Commit 26c8909.
- `encoder_encoder_pcn.py` — `self._infonce_codes=(inter2,inter12)`; inert, NATIVE
  `GATE_MATCH nlayers=143` verified. Commit d193df3.
- `train_coco64.py` — opt-in eager InfoNCE path (`--infonce-lambda/--infonce-tau`);
  inject at inter2/inter12 each relax substep, then the usual local LARS weight step;
  logs retrieval accuracy. lambda=0 keeps the old path. Commit 091420c.

PC-native invariant independently verified at each model-touching step: contrastive
gradient touches codes only, injected into `.state` only, weights learned solely by
`update_wts()/update_b()` (local LARS), correct descent sign, codes are the distinct
per-modality bottlenecks upstream of the shared-state alias.

## Experiment: COCO64 2k overfit, lambda sweep (JOBs 8421/8423/8424, H200)

Config: pairs 2000, lr 1e-3, weight_decay 3e-2, state_clip 400, gelu conv, relax 15,
batch 8, tau 0.07. Swept `--infonce-lambda` ∈ {0.1, 0.3, 1.0}.

Reconstruction converged to its floor by step 150 in every run (energy ≈ 0.005–0.008,
max|state| ≈ 42–59, no divergence). The alignment signal (`infonce_loss`, batch
retrieval `infonce_acc`):

| lambda | step 150 | step 300–350 | step 500 |
|--------|----------|--------------|----------|
| 0.1 | acc 0.750 / loss 2.298 | acc 0.375 / loss 2.135 | acc 0.125 / loss 2.125 |
| 0.3 | acc 0.750 / loss 2.074 | acc 0.375 / loss 2.077 | acc 0.500 / loss 2.079 |
| 1.0 | acc 0.625 / loss 2.071 | acc 0.500 / loss 2.078 | acc 0.750 / loss 2.077 |

`infonce_loss` is **pinned at chance** (ln 8 = 2.079) across the entire 10× lambda
range — it never descends. `infonce_acc` bounces around low values (batch=8 is coarse)
and never rises confidently; at lambda 0.1 it decayed to the 0.125 chance floor. So the
image and text codes stay at chance alignment regardless of coupling strength.

**Mechanism.** Reconstruction converges to a sharp equilibrium (energy at floor). The
injected InfoNCE nudge on inter2/inter12 is pulled back toward that recon equilibrium by
each subsequent `update_state`, so the relaxed state barely shifts toward alignment →
the local LARS weight step learns ~no alignment → the next forward pass still produces
chance-aligned codes. A self-consistent chance equilibrium that lambda magnitude alone
does not break (and stronger lambda stayed stable, so this is not a stability limit).

## text→image retest (150 relax steps, in-sample pairs)

Ran the same probe on the strongest-coupling checkpoint (`ckpt_infonce_l10_best`,
lambda 1.0) AND on the no-InfoNCE baseline (`ckpt_gelu_best`):

- **text→image** (image inferred from ZERO init, caption clamped): **uniform blob**,
  91-byte all-zero PNG, std 0.00000, for every pair — in BOTH the InfoNCE model and the
  baseline. Text provides zero top-down drive to the image. Visually confirmed (pure
  black square).
- **reconstruction** (image init, relax): exact — recon PNG matches the true image
  pixel-for-pixel (std 0.10–0.22, real structure). The image path is healthy in both.
- **image→caption**: my quick probe's caption readout collapses to a constant string
  IDENTICALLY in both the InfoNCE and baseline models → a probe artifact (zeros-init for
  the generated caption vs a warm init), NOT an InfoNCE regression. This probe does not
  re-establish the working image→caption direction; it only shows InfoNCE did not change
  it relative to baseline.

## Verdict

**PC-native InfoNCE state-injection at the deepest shared code does NOT unblock
text→image.** Across lambda 0.1/0.3/1.0 the code alignment never rose above chance
(`infonce_loss` pinned at ln 8), and text→image remained a uniform blob — indistinguishable
from the no-InfoNCE baseline. Reconstruction stays intact. The image-dominated latent is
unmoved. This is consistent with the functional version's experience (InfoNCE + warmup +
scale could not crack cross-modal generation) and with the plan's stated risk that
coupling the codes may be necessary-but-not-sufficient. Here the coupling did not even
take hold: the reconstruction equilibrium dominates the relaxation and the local weight
learning, so the injected contrastive error cannot reshape the codes.

## Why (interpretation for the paper)

The two branch codes both feed a single shared latent. When reconstruction has converged,
the states sit in a sharp recon basin; a batch-global contrastive error added to the
per-example relaxation is a small, transient perturbation that the recon dynamics undo
before the weight step reads the state. So the mechanism that makes PC weight-learning
work (learn from the relaxed equilibrium) is exactly what prevents the InfoNCE nudge from
being learned. Making alignment stick likely needs the coupling to shape the equilibrium
itself, not perturb it — e.g. an alignment term inside the energy the states relax to (so
recon and alignment reach a joint fixed point), a warm-up that aligns codes before recon
hardens, or contrasting a code that is not pinned by the shared-state alias. These are
follow-ups, not this scope.

## Artifacts

- Checkpoints (cluster): `ckpt_infonce_best` (λ0.1), `ckpt_infonce_l03_best` (λ0.3),
  `ckpt_infonce_l10_best` (λ1.0); logs `infonce_8421.log`, `infonce_l03_8423.log`,
  `infonce_l10_8424.log`.
- Retest images (local): `infonce_gen_l10/` (λ1.0), `infonce_gen/` (gelu baseline) —
  `t2i_*` (all black), `recon_*`/`true_*` (match), `results.json`.
- Throwaway probes `tools/infonce_gen_retest.py` / `tools/run_infonce.sh` are launchers,
  not committed.
