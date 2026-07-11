# Design: PC-native weight normalization to stabilize norm inflation

Date: 2026-07-11

## Goal

Stop the norm-inflation instability that caps every generative-training approach at ~ep13,
by reparameterizing each opt-in layer's weight as a controlled magnitude times a unit
direction, so the weight norm can no longer run away. Then re-run the CHL objective long
enough for the decode to sharpen. Entirely within bidirectional predictive coding.

## Background (the instability)

The class's beta-less LARS weight step is `wts -= lr * trust * (g_w + wd*wts)`, with
`trust = ||wts|| / (||g_w|| + wd*||wts|| + 1e-6)`. As `||wts||` grows, trust grows, so the
update grows, so `||wts||` grows faster — a positive feedback. Growing weights inflate the
forward activations; the states climb, hit `state_clip=400`, and the energy explodes. Weight
decay only delays it (recon-only creeps to ~ep25; the generative step accelerates it to
~ep13). Every generative objective (InfoNCE, bridge, gentle-lr, CHL) hits this same wall.
The root is the weight norm growing freely and feeding the trust ratio.

## Core: weight normalization

Reparameterize each opt-in layer's weight as `w = g * v/||v||`, a direction `v` (the current
weight tensor) and a magnitude `g`, PER OUTPUT UNIT (standard weight-norm). Because
`||w|| = |g|` (per unit), the magnitude is a controlled learned scalar rather than free to
inflate. The direction `v` is updated only tangentially (so `||v||` is ~preserved and cannot
norm-inflate), and `g` is damped so `||w||` stays bounded. This removes the runaway term at
its source and lets us DROP the LARS trust ratio for these layers (the exact term that grew
with `||w||`).

Shapes. Dense `wts` is `(in, out)`, normalize over axis 0 (per column/output), `g` is
`(out,)`. Conv `wts` is `(kh, kw, in, out)`, normalize over axes (0,1,2) (per output filter),
`g` is `(out,)`.

## This is bidirectional PC

- `w = g*v/||v||` is the SAME shared weight used both directions: `predict_next` (encode, up)
  and `predict_prev` (decode, down) both use it. One net, no separate decoder.
- The update is a change of variables on the class's OWN local gradient, no backprop. The
  existing `update_wts` computes the local weight gradient `g_w = (d_state + d_pred)/denom`
  from local prediction errors. Weight-norm splits `g_w` into a radial part (updates `g`) and
  a tangential part (updates `v`). No global loss, no optimizer, no change to how `g_w` is
  computed.

## Component 1: the reparameterization and the `weight()` accessor

Add to `Conv2DPCNLayer` and `DensePCNLayer`:
- a `weight_norm` flag (default False) and a magnitude variable stored as `self.g_mag`
  (named to avoid colliding with the many `g` locals in `update_wts`), created when
  weight-norm is enabled and initialized to the current per-unit norm of `wts` so `w == wts`
  at enable time, seamless for a warm start.
- a `weight()` method: returns `self.wts` when `weight_norm` is False (byte-identical to
  today), else `g_mag * wts / ||wts||` with the per-unit norm broadcast over the input axes.

Replace every DIRECT use of `self.wts` in the forward/decode/state computations with
`self.weight()`. Uses that already route through `net_in`/`predict_next`/`predict_prev` need
no further change, since those methods now call `weight()`; the plan reads the code to see
which is which. The direct sites to expect:
- Dense: `net_in` (`x @ weight() + b`), `predict_prev` (`(state-b) @ weight()^T`),
  `pred_loss_d_input` (`... @ weight()^T`), and any direct `self.wts` term in `update_state`.
- Conv: `net_in` (`conv2d(x, weight(), ...)`), `predict_prev` (`conv2d_transpose(state,
  weight(), ...)`), `pred_loss_d_input` (transpose with `weight()`), and any direct
  `self.wts` term in `update_state`.
Do NOT change the `self.wts.shape` references used only for sizing (e.g. `filter_sizes`), and
keep the `weight_norm=False` path returning `self.wts` unchanged so NATIVE is byte-identical.

## Component 2: the weight-norm update (radial/tangential split of the local gradient)

In `update_wts`, when `weight_norm` is True, reuse the SAME local weight gradient the class
already forms just before its LARS trust factor (the combined `d_state + d_pred` term, in the
shape of `wts`); call it `g_w`. It is now the gradient w.r.t. the effective weight `w`, since
the forward/decode used `weight()`. Then, instead of the LARS step, do the standard
weight-norm decomposition, per output unit:
- `vhat = wts / ||wts||`  (per-unit unit direction, norm over the input axes)
- radial `dg = sum(g_w * vhat)` over the input axes  (shape `(out,)`)
- tangential `dv = (g_mag/||wts||) * (g_w - dg*vhat)`  (broadcast; shape of `wts`, ⊥ to `vhat`)
- `g_mag.assign_sub(lr * (dg + wd*g_mag))`   (magnitude, damped by wd so `||w||` is controlled)
- `wts.assign_sub(lr * dv)`                  (direction; tangential, so `||wts||` ~preserved)
No LARS trust ratio for weight-norm layers. `lr`/`wd` are the layer's `learning_rate`/
`weight_decay` (so the existing recipe and the `--gen-lr`/CHL sign-flip still apply through
`learning_rate`, flipping both the `g_mag` and `wts` updates together). The `g_mag` damping
that bounds `||w||` comes from the recon steps, which keep `wd` on; the CHL contrast steps run
`wd=0` as today. `update_b` is unchanged.

## Component 3: opt-in via a training flag

Add `--weight-norm` to `train_coco64.py` (default off). When set, after building the model
and realizing weights, enable `weight_norm` on the conv/dense trainable layers (initializing
each `g` to the current per-unit `||wts||`). NATIVE and every run without `--weight-norm` are
byte-identical. This composes with `--train-mode chl` and the warm start.

## Component 4: re-run CHL with weight-norm

Warm-start from the recon-trained COCO64_GEN checkpoint, enable `--weight-norm`, run
`--train-mode chl`, and check that training now holds WELL PAST ep13 without the state hitting
the clip, so the decode trains long enough. Fix the Monitor B-gate first to watch
`max_abs_state` near the clip and energy climbing (the earlier gap).

## Validation

- NATIVE `GATE_MATCH nlayers=143` with `--weight-norm` OFF (the `weight()` accessor returns
  `self.wts` and the LARS path is unchanged, so NATIVE is byte-identical). This proves the
  reparameterization is inert when off.
- A local unit test: with `weight_norm` on and `g_mag` init to the per-unit norm, `weight()`
  equals the original `wts` (seamless enable); a tangential `dv` step leaves `||wts||` ~fixed;
  and `||weight()|| == |g_mag|` per unit.
- Stability run on COCO64_GEN with `--weight-norm` (recon or chl): does `max_abs_state` stay
  bounded and NOT climb to the clip well past ep13, versus the un-normalized run? This is the
  make-or-break check.
- Then the CHL retrain + darkness diagnostic + text-to-image retest: with training holding
  longer, does the decode brighten (gen/true toward 1.0) and sharpen toward caption-specific
  structure, recon/i2t intact.

## Risks and unknowns

- Correctness of the radial/tangential decomposition (the standard weight-norm gradient);
  the local unit test guards the seamless-enable and norm-preservation properties.
- Whether weight-norm alone fully stabilizes; if `g_mag` still drifts, the fallback is to FIX
  `g_mag` at init (learn direction only), the most robustly stable variant.
- The accessor touches gate-critical layer methods; the `weight_norm=False` path must stay
  byte-identical (NATIVE gate is the guard).
- Dropping the LARS trust for weight-norm layers may change the learning dynamics; the
  stability run and the CHL retest are the checks.
- Weight-norm stabilizes training, but recognizable generation still depends on the CHL
  objective working over the longer horizon; this fixes the wall, not necessarily the ceiling.

## Out of scope

muP and the state-norm-penalty alternatives (this chose weight-norm), the fix-`g_mag` variant
(fallback only), held-out, and the capacity ladder.
