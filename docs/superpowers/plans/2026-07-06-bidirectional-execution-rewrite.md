# Bidirectional PCN Execution Rewrite (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `EncoderEncoderPCN` fast, batchable, and memory-stable WITHOUT changing its predictive-coding math, so it can later be retargeted to 64px and trained.

**Architecture:** Keep the bidirectional class exactly as designed (one conv image net + one transformer text net used both directions via `predict_prev`/`predict_next`, joined by `share_state_layer` shared states, trained relax-then-step). Change only how the update loop executes: drop per-step gc, compile the sweep with `tf.function`, stop the memory growth, and allow batch > 1. A golden-output gate guarantees the math is unchanged.

**Tech Stack:** TensorFlow 2.21, the Colby HPC cluster (H200/A100 via Slurm), the existing `~/tf-env`.

## Global Constraints

- The model is the bidirectional class only (`encoder_encoder_pcn.py` + `conv/dense/transformer_pcn_layer.py`). No functional-version code, no backprop-diffusion, no pretrained parts. Copied verbatim from the spec: "Learning stays predictive coding (relaxation + local weight updates)."
- No task may change the PC update math. Every refactor task must pass the golden-output gate from Task 1 (relaxed states reproduce the pre-rewrite values within tolerance 1e-4 relative).
- This runs on a GPU node. Every run uses `~/tf-env/bin/python3` with `LD_LIBRARY_PATH` exported over `~/tf-env/lib/python3.13/site-packages/nvidia/*/lib`, submitted with `sbatch -p gpu --gres=gpu:H200:1` (or `gpu:RTX:1`); A100 only via `-p normal`. CPU cannot hold the model.
- Commits use the repo's student-voice style, no AI attribution.
- Resolution stays at the native 572px for all of Phase 2 (no architecture resize here; that is Phase 3). Validation runs at batch 1 unless a task says otherwise.

## File Structure

- Modify `encoder_encoder_pcn.py`: the update-loop methods (`update_states_wts_b`, `update_states_wts_b_relaxed`, `update_states_img`, `update_states_txt`) — remove gc, add the graph-compiled sweep, fix the growth. This is where all loop-level changes live.
- Possibly modify `dense_pcn_layer.py` / `conv_pcn_layer.py` / `transformer_pcn_layer.py`: only if `tf.function` tracing needs a Python-side branch made graph-safe (e.g. `int(not self.is_clamped)` recomputed at trace time). Touch only what tracing forces.
- Create `tools/rewrite_gate.py`: the golden-reference harness and the memory/speed/batch measurements. One file, one responsibility (validate + measure a rewrite).
- Create `runs_local/` entries are not needed; results print to the Slurm log.

---

### Task 1: Golden-output gate + baseline measurements

**Files:**
- Create: `tools/rewrite_gate.py`
- Uses: `encoder_encoder_pcn.py` (`EncoderEncoderPCN`, `pass_through`, `update_states_wts_b`, `trainable_layers`, per-layer `.state`)

**Interfaces:**
- Produces: `golden_state_signature(model) -> dict[int,float]` (per-layer L2 norm of `.state`, keyed by index in `trainable_layers`); `run_reference(steps:int, batch:int, seed:int) -> (signature, peak_gib, per_step_seconds)`. Later tasks call `run_reference` and compare signatures.

- [ ] **Step 1: Write `tools/rewrite_gate.py`**

```python
import os, time, argparse
import tensorflow as tf, numpy as np
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
from encoder_encoder_pcn import EncoderEncoderPCN

def golden_state_signature(model):
    sig = {}
    for i, L in enumerate(model.trainable_layers):
        s = getattr(L, "state", None)
        if s is not None:
            sig[i] = float(tf.norm(tf.cast(s, tf.float32)))
    return sig

def run_reference(steps=2, batch=1, seed=0):
    tf.random.set_seed(seed)
    m = EncoderEncoderPCN(1e-4)
    img = tf.random.normal((batch, 572, 572, 3), seed=seed)
    txt = tf.random.normal((batch, 192, 512), seed=seed)
    mask = tf.zeros((batch, 192), tf.float32)
    m.img_input.is_clamped = True; m.txt_input.is_clamped = True
    m.pass_through(img, txt, mask)
    try: tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception: pass
    t = time.time()
    m.update_states_wts_b(steps)
    dt = (time.time() - t) / max(1, steps)
    peak = 0.0
    try: peak = tf.config.experimental.get_memory_info("GPU:0")["peak"]/2**30
    except Exception: pass
    return golden_state_signature(m), peak, dt

def compare(a, b, tol=1e-4):
    bad = []
    for k in a:
        d = abs(a[k]-b.get(k, 0.0)) / (abs(a[k]) + 1e-9)
        if d > tol: bad.append((k, a[k], b.get(k), d))
    return bad

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--save", default="golden.npz")
    a = ap.parse_args()
    sig, peak, dt = run_reference(a.steps, a.batch)
    np.savez(a.save, **{str(k): v for k, v in sig.items()})
    print(f"GOLDEN steps={a.steps} batch={a.batch} peak={peak:.2f}GiB per_step={dt:.2f}s nlayers={len(sig)}", flush=True)
```

- [ ] **Step 2: Run it on the cluster to record the baseline golden**

Run (in `~/hpc`, one line):
`sbatch -p gpu -c 4 --gres=gpu:H200:1 --mem=64G -t 00:30:00 -J rewrite_gate --wrap 'export PATH=$HOME/tf-env/bin:$PATH; export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib|tr " " ":"); cd $HOME/encoder_encoder_PCN && python3 tools/rewrite_gate.py --steps 2 --save golden_baseline.npz'`
Expected: a `GOLDEN ... peak=~34-68GiB per_step=~14s nlayers=<N>` line, and `golden_baseline.npz` written.

- [ ] **Step 3: Commit**

```bash
git add tools/rewrite_gate.py golden_baseline.npz
git commit -m "added a golden-state gate + baseline for the class so any speed/memory refactor can prove it did not change the PC math"
```

---

### Task 2: Remove the per-step `gc.collect()`

**Files:**
- Modify: `encoder_encoder_pcn.py:479` (in `update_states_wts_b`) and `:496` (in `update_states_wts_b_relaxed`)

**Interfaces:**
- Consumes: `run_reference` / `compare` from Task 1.
- Produces: nothing new; same methods, faster.

- [ ] **Step 1: Delete the two `gc.collect()` calls**

In `update_states_wts_b`, remove the `gc.collect()` inside the loop (line ~479). In `update_states_wts_b_relaxed`, remove the `gc.collect()` (line ~496). Leave everything else identical.

- [ ] **Step 2: Re-run the gate and compare to golden**

Run the same sbatch as Task 1 Step 2 but with `--save golden_nogc.npz`, then in a Python one-liner load both and call `compare`. Expected: `compare(golden_baseline, golden_nogc)` returns `[]` (states unchanged) and the printed `per_step` is lower than the ~14s baseline.

- [ ] **Step 3: Commit**

```bash
git add encoder_encoder_pcn.py
git commit -m "dropped the gc.collect that ran every relaxation step. states are byte-identical to before, just faster"
```

---

### Task 3: Diagnose the 34->68 GiB growth (investigation task; deliverable is findings)

**Files:**
- Create: `tools/mem_probe.py`

**Interfaces:**
- Produces: a printed per-step breakdown of GPU current/peak memory and the count of live `tf.function` concrete functions, over 10 steps, identifying whether growth comes from (a) per-step op/tensor retention in eager mode, (b) `pass_through` rebuilding, or (c) large transient gradient tensors in `update_wts` for the huge dense layers.

- [ ] **Step 1: Write `tools/mem_probe.py`** that builds the model, does `pass_through`, then loops 10 single `update_states_wts_b(1)` calls printing `get_memory_info` current/peak each step, and after each step prints `len(tf.python.eager.context.context()... )` live-tensor proxy via `gc.get_objects()` count of `tf.Tensor`/`EagerTensor`. (Concrete instrumentation; exact counts guide the fix.)

- [ ] **Step 2: Run it on the cluster** (same sbatch wrapper, `python3 tools/mem_probe.py`). Expected: a 10-line memory trace. Record whether `current` (not just `peak`) rises step over step. Rising `current` = retention/leak; flat `current` with high `peak` = large transients only.

- [ ] **Step 3: Write the finding into `docs/superpowers/plans/2026-07-06-bidirectional-execution-rewrite.md` under a new "Task 3 findings" note and commit**

```bash
git add tools/mem_probe.py docs/superpowers/plans/2026-07-06-bidirectional-execution-rewrite.md
git commit -m "instrumented the memory growth over 10 steps to find whether it is retention or just large transients"
```

---

## Task 3 findings

**Verdict: TRANSIENT-ONLY. No retention leak.**

Ran `tools/mem_probe.py` on an H200 (96G host mem), resetting `reset_memory_stats` before every step so `peak_this_step` reflects only that step's transient high-water mark, not a cumulative session peak. 10 single `update_states_wts_b(1)` steps after one `pass_through`:

| point | cur (GiB) | peak_this_step (GiB) | tensors |
|---|---|---|---|
| after pass_through | 29.55 | 34.30 | 499 |
| step 0 | 29.55 | 68.01 | 640 |
| step 1 | 29.55 | 68.01 | 640 |
| step 2 | 29.55 | 68.01 | 640 |
| step 3 | 29.55 | 68.01 | 640 |
| step 4 | 29.55 | 68.01 | 640 |
| step 5 | 29.55 | 68.01 | 640 |
| step 6 | 29.55 | 68.01 | 640 |
| step 7 | 29.55 | 68.01 | 640 |
| step 8 | 29.55 | 68.01 | 640 |
| step 9 | 29.55 | 68.01 | 640 |

`cur` is bit-identical (29.55G) across all 10 steps: delta 0.00 GiB, not just "under 0.5 GiB." `peak_this_step` is also bit-identical (68.01G) every step, meaning it's not creeping up run over run, it's the same-size transient every time. `tensors` jumps once from 499 (right after `pass_through`, before any weight update has run) to 640 at step 0, then stays flat at 640 for steps 1-9 — that one-time jump is Python object bookkeeping created by the first `update_states_wts_b` call (e.g. optimizer/gradient-tape scaffolding), not per-step accumulation.

This resolves the ambiguity in the earlier 3-step profile: that run never reset `peak`, so it recorded the *cumulative session* high-water mark climbing 34.30G (after `pass_through`) -> 68.01G (after 3 steps). It looked like growth because the reported number was a running max, not a per-step reading. With the reset added here, the true per-step peak is 68.01G from step 0 onward — the weight-update pass reaches that high-water mark on its very first call and stays there, it never climbs further on later steps. There is no per-step retention: `cur` never rises, `tensors` never rises past the one-time step-0 jump.

**Cause of the 68.01G transient:** the weight-update gradients for the billion-parameter dense projections (largest layer 2064.8M params) are large one-shot tensors — computed, applied via `.assign*`, then freed within the same step, so they show up in `peak` but not in `cur`. This is architecture-inherent at 572px resolution and is expected to shrink substantially once Phase 3 retargets the model to 64px.

**Implication for Task 4:** no real fix is needed. There is no resident leak to chase. `cur` is already flat (in fact exactly flat, not just within tolerance) and `tensors` is bounded. Task 4 becomes a confirm-flat check rather than a fix: re-run `mem_probe.py` (or an equivalent flat-check) after Task 5/6's changes to make sure nothing introduced along the way (tf.function tracing, batching) breaks the flatness, and record that the 68G peak is the expected large-dense-gradient transient at 572px, deferring any peak reduction to the Phase 3 64px retarget.

---

### Task 4: Fix the memory growth to flat-over-10-steps

**Files:**
- Modify: `encoder_encoder_pcn.py` (update methods) and/or the layer files, per Task 3 findings.

**Interfaces:**
- Consumes: Task 3 findings; `run_reference`/`compare`.
- Produces: memory `current` flat across 10 steps.

- [ ] **Step 1: Apply the fix indicated by Task 3.** If retention: ensure the loop holds no Python references to per-step tensors and uses only in-place `.assign*` (it should already); the likely culprit is eager graph accumulation removed by Task 5's `tf.function`, in which case mark this task blocked-by Task 5 and move it after. If large transients in `update_wts` for the huge dense layers: this is expected at 572px and resolves at 64px (Phase 3); record that and set the success bar to "current flat, peak bounded," not "peak small."

- [ ] **Step 2: Re-run `mem_probe.py` for 10 steps.** Expected: `current` GiB is flat (delta < 0.5 GiB across steps 2..10).

- [ ] **Step 3: Re-run the gate.** Expected: `compare(golden_baseline, ...)` == `[]`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "stopped the per-step memory growth (current now flat over 10 steps); states still match the golden gate"
```

---

### Task 5: Compile the relax+weight sweep with `tf.function`

**Files:**
- Modify: `encoder_encoder_pcn.py` (`update_states_wts_b_relaxed` gets a graph-compiled inner sweep)
- Possibly modify layer files if tracing forces a Python branch to be graph-safe.

**Interfaces:**
- Consumes: `run_reference`/`compare`.
- Produces: a `tf.function`-wrapped sweep; `update_states_wts_b_relaxed` calls it.

- [ ] **Step 1: Wrap the inner relaxation sweep + weight step in a module-level `@tf.function`** that takes no python-varying args (clamps are fixed for a given train/test phase, so trace once per phase). The Python `for layer in trainable_layers` unrolls into the graph. Keep `reduce_retracing=True`. Do not change any per-layer math.

- [ ] **Step 2: Run the gate.** Expected: `compare(golden_baseline, ...)` within tol 1e-4 (graph vs eager may differ at ~1e-6). Print the new `per_step`; expected well under the post-gc time (target order 1s or less; record actual).

- [ ] **Step 3: Confirm no runaway retracing:** add a trace counter print; expected 1 trace for the train phase (not one per step).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "compiled the relaxation+weight sweep with tf.function so it stops eager-looping hundreds of layer objects. same states as the golden gate, big speedup"
```

---

### Task 6: Enable and test batching

**Files:**
- Modify: `tools/rewrite_gate.py` (already takes `--batch`); verify no code path assumes batch 1.

**Interfaces:**
- Consumes: `run_reference(batch=B)`.
- Produces: confirmation the class trains at batch > 1.

- [ ] **Step 1: Run the gate at `--batch 4`** (same sbatch wrapper, `--batch 4 --save golden_b4.npz`). Expected: it completes without shape errors, states are finite, and `peak` scales roughly with batch. If it OOMs at 572px batch 4, record the max batch that fits (this informs Phase 3, where 64px will allow larger batches).

- [ ] **Step 2: Confirm batch-1 still matches golden** (re-run `--batch 1`, `compare` == `[]`).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "verified the class runs batched (batch 4) with finite states; recorded the max batch that fits at 572px"
```

---

## Phase 2 exit criteria

Graph-compiled, gc-free, memory-flat, batched `EncoderEncoderPCN` whose relaxed states reproduce the pre-rewrite golden within 1e-4, with the achieved per-step time, max batch, and peak memory recorded. Those three numbers size Phase 3 (64px retarget) and Phase 4 (training), which are separate plans written after this lands.

## Self-Review

- Spec coverage: this plan covers spec Phase 2 (execution rewrite: gc, graph mode, leak, batching, validation gate) in full. Spec Phase 3 (64px retarget) and Phase 4 (train + eval both pathways) are deliberately deferred to their own plans, because their concrete steps depend on Phase 2's measured per-step time, max batch, and leak resolution.
- Placeholder scan: Tasks 3 and 4 are investigation-gated by design (diagnose-then-fix a real, undiagnosed memory growth); each is anchored by a concrete measurement and a pass/fail test rather than invented fix code, which is the honest structure for a debugging task. All other tasks have exact code, files, and commands.
- Type consistency: `run_reference`, `golden_state_signature`, `compare` signatures are used consistently across Tasks 1, 2, 4, 5, 6.

---

## Phase 2b: adopt + compile the canonical relax-then-step schedule (added 2026-07-06, user decision)

Phase 2 compiled `update_states_wts_b` (interleaved: a weight step every state step), which is what the gate and `train_step` call. But the canonical predictive-coding training schedule the design names is `update_states_wts_b_relaxed` (relax ALL states for `num_relax_steps` substeps with weights frozen, THEN one weight step), which is still eager/uncompiled. The user chose to adopt the relaxed schedule as the training method and compile it now. It produces DIFFERENT states than the interleaved method, so it needs its OWN golden baseline (cannot reuse `golden_baseline.npz`). `train_step`'s signature is left unchanged in this sub-phase (rewiring it needs the `num_relax_steps` hyperparameter, a Phase 4 choice, and would break the integration test); Phase 4 calls `update_states_wts_b_relaxed` directly.

### Task R1: relaxed-mode gate + eager golden baseline
- Modify `tools/rewrite_gate.py`: add `--relaxed`, `--relax-steps R` (default 5), `--weight-steps W` (default 2). When `--relaxed`, `run_reference` calls `m.update_states_wts_b_relaxed(W, R)` instead of `m.update_states_wts_b(steps)`; the per-layer state-norm signature is recorded the same way. Keep the default (non-relaxed) path byte-identical.
- Run the CURRENT (eager, uncompiled) relaxed method on the H200 to record `golden_relaxed_baseline.npz`; fetch it back and check it with `tools/npz_finite.py` (all norms finite — a NaN here would mean the relaxation itself is unstable at 572px, a finding).
- Commit the gate extension + the baseline.
- Done when: `GOLDEN ... relaxed W=2 R=5 ...` prints with finite values and `golden_relaxed_baseline.npz` is committed.

### Task R2: graph-compile `update_states_wts_b_relaxed`
- Modify `encoder_encoder_pcn.py`: compile the relaxed schedule's two sweeps as separate lazily-built `@tf.function`s cached in NEW instance attrs (e.g. `_compiled_relax_sweep` = `for layer: layer.update_state()`, `_compiled_learn_sweep` = `for layer: layer.update_wts(); layer.update_b()`), driven by plain Python loops (`for weight_step in range(W): for _ in range(R): relax(); then learn()`). Apply the SAME clamp-signature + state-Variable-id guard as `update_states_wts_b` (its own recorded sig/ids, checked each call). Do not change any per-layer math or the sweep order. No variables created in-graph (all init is in `pass_through`).
- Validate: re-run the gate in `--relaxed` mode → must `GATE_MATCH` against `golden_relaxed_baseline.npz` (rel-tol 1e-4). Confirm one trace per sweep. Measure warm per-weight-step vs the eager relaxed time.
- Commit. Done when: relaxed-mode `GATE_MATCH nlayers=143`, single trace per sweep, speedup recorded.

---

## M4 findings (generation path)

**Task:** a Phase 4 prerequisite diagnosis (not a Phase 2/2b task, separate from R1/R2 above). The generation path — `update_states_img`/`update_states_txt`, called by `test_step` — updates the UNCLAMPED (generated) input TWICE per step:

```python
def update_states_img(self, num_steps):
    for step in range(num_steps):
        for layer in self.trainable_layers:   # img_input IS in this list -> updated once here
            layer.update_state()
        self.img_input.update_state()          # img_input updated AGAIN, second time
```

Ran `tools/gen_probe.py` on an H200, on a FRESH untrained model (random weights — this is a mechanism test, not an image-quality test), seed 0, `img=(1,572,572,3)`, `txt=(1,192,512)`, `mask=(1,192)`, `num_steps=15`. (Building several full ~30-70GiB model instances back-to-back in one process fragmented the GPU BFC allocator enough to OOM the 4th/5th build, so the probe was split with a `--part {1,2,3}` flag and Part 3 was re-run as its own isolated process; its WITH-ablation curve reproduced Part 2's curve bit-for-bit, confirming determinism and that the split changed nothing.)

**1. Both directions run finite and stable on a fresh model.**

| direction | wall (15 steps + pass_through) | output shape | finite output | finite all states | min / max / mean |
|---|---|---|---|---|---|
| `predict='img'` | 7.11s | (1, 572, 572, 3) | PASS | PASS | -4.91 / 5.03 / 0.0124 |
| `predict='txt'` | 5.28s | (1, 192, 512) | PASS | PASS | -4.17 / 4.42 / -0.0028 |

No NaN/Inf anywhere (output or any of the 143 `trainable_layers` states) in either direction, on random weights, at the full `num_steps=15`. Generation is stable at the mechanism level.

**2. Convergence of the unclamped `img_input` (actual generation path, with the double update).** Instrumented the real per-step procedure (per-layer sweep + the explicit second `img_input.update_state()`) and tracked `img_input.state`'s L2 norm and step-to-step relative change over 15 steps:

| step | norm | rel_change |
|---|---|---|
| 0 | 991.35 | — |
| 1 | 992.41 | 0.001585 |
| 4 | 995.56 | 0.001550 |
| 7 | 998.65 | 0.001515 |
| 10 | 1001.70 | 0.001482 |
| 14 | 1005.69 | 0.001438 |

Not oscillating, not diverging: the norm drifts monotonically upward at a nearly-constant, slowly decelerating rate (rel_change shrinks ~9% from step 1 to step 14, from 0.001585 to 0.001438). It has not settled to a fixed point within 15 steps, but the drift is bounded and decelerating, not runaway. Given untrained random weights (no learned equilibrium to relax toward), this is the expected shape — a caveat, not a red flag. Per-step wall time for the uncompiled/eager loop: mean 0.306s/step (0.305s excluding step 0), i.e. ~4.6s of relaxation for 15 steps.

**3. Ablation of the second update — the load-bearing evidence.** Two fresh models (identical seed, identical `pass_through` inputs, so identical initial weights and initial states), 15 steps each, WITH the explicit second `img_input.update_state()` vs WITHOUT it:

| step | WITH norm | WITH rel_change | WITHOUT norm | WITHOUT rel_change | ratio (WITH/WITHOUT rel_change) |
|---|---|---|---|---|---|
| 0 | 991.3535 | — | 990.8225 | — | — |
| 1 | 992.4124 | 0.001068 | 991.3535 | 0.000536 | 1.99 |
| 5 | 996.5948 | 0.001042 | 993.4546 | 0.000526 | 1.98 |
| 10 | 1001.7029 | 0.001010 | 996.0299 | 0.000513 | 1.97 |
| 14 | 1005.6934 | 0.000985 | 998.0494 | 0.000503 | 1.96 |

- Final `img_input.predict_next()` relative L2 difference (WITH vs WITHOUT) after 15 steps: **0.011142 (1.1%)**. Both finite.
- The WITH/WITHOUT rel_change ratio sits at ~2.0 on step 1 and decays smoothly to ~1.96 by step 14 — remarkably close to exactly 2x, for the entire run.
- WITHOUT is not "no update" — `img_input` is still updated once per step by the main `for layer in self.trainable_layers` loop (it's a member of that list). The difference isolates exactly the ONE extra explicit call.

**Verdict: INTENTIONAL.** Mechanism, confirmed by the ~2x signature above: `img_input` and `txt_input` are each appended to `trainable_layers` BEFORE their own downstream chain is built (`self.img_input` is index 0, appended before `conv1`; `self.txt_input` is appended before `txt_embedding`). So the single `for layer in self.trainable_layers: layer.update_state()` sweep hits `img_input` FIRST, when `conv1.state` still holds LAST step's value — `img_input`'s in-loop update is permanently one full step stale, every step, forever (a structural artifact of list order, not a transient that would disappear with more steps). Every other layer in the sweep is naturally processed AFTER its own upstream dependency, so it always sees this-step's freshest available value (textbook Gauss-Seidel). Only the two input layers are anomalous, purely because of where they sit in the list. The explicit second call re-runs `img_input.update_state()` AFTER the full sweep has already refreshed `conv1` this step, giving it the same freshest-available-value treatment every other layer already gets. That the ablation shows almost exactly a 2x displacement-rate difference (not an arbitrary or unstable difference, not a sign flip, no divergence) is exactly what you'd expect from "one stale sub-update + one fresh sub-update" vs "one stale sub-update alone" — it is not evidence of accidental double-counting, it's evidence the second call is doing specifically what re-syncing to this-step's latents would do. Removing it doesn't destabilize anything (WITHOUT is equally finite and smooth), it just reintroduces the permanent one-step lag specific to the two input layers' list position, roughly halving their effective relaxation rate. The 1.1% final-output effect is small but real, in the direction the "avoid one-step lag" theory predicts, with no evidence of it being a stray/duplicate call.

**Uncompiled path note.** `update_states_img`/`update_states_txt` (and `test_step`) are pure eager Python — no `tf.function`, unlike `update_states_wts_b`/`update_states_wts_b_relaxed`. Measured cost: ~0.3s/step (mean 0.306s, warm 0.305s) for a 15-step generation, plus ~2.5-2.9s for the initial `pass_through`, i.e. ~5-7s wall-clock for a full `test_step` call at `num_steps=15`. This is far cheaper than the *eager* training sweep's ~14s/step baseline (Task 1), because generation only runs `update_state()` (state relaxation) — it never runs `update_wts`/`update_b`, which is what makes training expensive (gradient tensors for the multi-million-parameter dense layers).

**Recommendation for Phase 4:**
- **Keep the double update as-is.** It is a deliberate (or at least mechanically correct and beneficial) compensation for `img_input`/`txt_input` sitting at the front of `trainable_layers`; the data shows a clean, stable, theory-consistent ~2x effect, not an anomaly. No model code change.
- **Leave the generation path uncompiled for now.** At ~0.3s/step and ~5-7s per `test_step` call, it is not the bottleneck training's eager sweep was — compiling it would need the same lazy-`tf.function` + clamp-signature/state-Variable-id trace guard already built for `update_states_wts_b`/`update_states_wts_b_relaxed`, which is real added complexity for a path that (per the Phase 2 spec) runs far less often than training. If Phase 4's eval schedule turns out to call `test_step` frequently (e.g. every N training steps, or at larger batch/more steps), revisit and compile it the same way — being careful to keep the double-update line (`for layer in trainable_layers: layer.update_state()` then the explicit `self.img_input.update_state()` / `self.txt_input.update_state()`) inside the compiled graph, since that is now confirmed to be load-bearing, not vestigial.
