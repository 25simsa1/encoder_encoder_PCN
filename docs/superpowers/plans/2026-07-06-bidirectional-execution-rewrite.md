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
