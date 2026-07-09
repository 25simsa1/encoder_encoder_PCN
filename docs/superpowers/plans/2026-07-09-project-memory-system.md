# Project Memory System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the encoder_encoder_PCN repo a persistent memory so Claude Code stops losing its instructions, conventions, and experiment state across and within sessions.

**Architecture:** A repo-native set of markdown files. `CLAUDE.md` at the root auto-loads every session and carries the invariants plus a session protocol. A `docs/` folder holds the living knowledge (STATE, an append-only experiment LOG, and an ARCHITECTURE map). Obsidian carries a dashboard note that links back to the repo and is only a window.

**Tech Stack:** Markdown, git. No code, no test framework. Verification is content checks plus a fresh-session smoke test.

## Global Constraints

- All prose follows the user's writing style. No em dashes, no colons in prose sentences.
- Commit messages are first-person and lowercase-leaning, no AI attribution, matching the repo's existing style (e.g. "added a downsample config field").
- Source of truth for project knowledge is the repo, never the Obsidian vault.
- The seeded content must be accurate to the codebase. When a fact is not certain, the implementer reads the named source file rather than guessing.
- `CLAUDE.md` stays short (aim under 60 lines) so it survives context compaction.

---

### Task 1: Root CLAUDE.md with invariants and session protocol

**Files:**
- Create: `CLAUDE.md`

**Interfaces:**
- Produces: the session protocol referenced by every later task, and the pointer to `docs/ARCHITECTURE.md`, `docs/STATE.md`, `docs/experiments/LOG.md`.

- [ ] **Step 1: Gather any extra invariants from history**

Run: `git log --oneline -40` and `sed -n '1,40p' CAPACITY.md`
Extract any hard rule stated as "must", "keep", "unchanged", "byte-identical", or a fixed experimental bar. Add each as a numbered invariant in Step 2 beneath the seed set already written there.

- [ ] **Step 2: Write CLAUDE.md**

```markdown
# CLAUDE.md, encoder_encoder_PCN

Research codebase for a predictive coding network (PCN) studying whether a cross-modal coupling failure persists as the model scales from 156M to 7.7B parameters. Read this fully before acting.

## Session protocol (every session)
1. At session start, read `docs/STATE.md` and the top three entries of `docs/experiments/LOG.md` before touching code.
2. After any run or work chunk, append a dated entry to the top of `docs/experiments/LOG.md` and update `docs/STATE.md`. Never edit past LOG entries.
3. Keep the invariants below intact. If a request conflicts with one, stop and say so.

## Hard invariants (do not violate)
1. The NATIVE config stays byte-identical. Its downsampling path is maxpool. Any change that alters NATIVE output is a regression, so guard it, do not touch it.
2. OOM is expected. The model is intentionally about 28.7 GiB and needs a GPU of at least 40GB. Never resolve an OOM by shrinking the model, cutting capacity, or dropping the documented batch. Fix the real bug instead.
3. The decisive experimental bar does not move. Coupling is judged at the 8k-pair scale with latent retrieval primary against the banked bar of more than 3 in 2000. Do not soften or redefine it.

## Configs
- NATIVE, the reference path, maxpool downsampling, must stay byte-identical.
- COCO64_GEN, strided-conv bidirectional downsampling so top-down generative drive flows.
- COCO64_156M, the 156M banked config.
Configs are defined in `pcn_config.py` and consumed by `train_coco64.py`.

## Orientation
Architecture and the file map live in `docs/ARCHITECTURE.md`. Current status lives in `docs/STATE.md`.
```

- [ ] **Step 3: Verify required content is present**

Run: `grep -c "NATIVE" CAPACITY.md CLAUDE.md; grep -q "Session protocol" CLAUDE.md && grep -q "Hard invariants" CLAUDE.md && echo OK`
Expected: prints `OK`, and CLAUDE.md is at most ~60 lines (`wc -l CLAUDE.md`).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "added a root CLAUDE.md so every session loads the invariants, configs, and a read-state protocol"
```

---

### Task 2: docs/ARCHITECTURE.md, the code map

**Files:**
- Create: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes: config names from `CLAUDE.md`.
- Produces: the file map that STATE and CLAUDE.md point to.

- [ ] **Step 1: Confirm the core modules exist and their roles**

Run: `head -30 encoder_encoder_pcn.py pcn_config.py train_coco64.py conv_pcn_layer.py dense_pcn_layer.py transformer_pcn_layer.py`
For each, read enough to write one accurate line describing its responsibility.

- [ ] **Step 2: Write docs/ARCHITECTURE.md**

Use this exact skeleton and fill each line from what Step 1 showed. Do not invent responsibilities, describe what the file actually does.

```markdown
# Architecture

## Core modules
- `encoder_encoder_pcn.py`, <the model definition, one accurate line>.
- `pcn_config.py`, config definitions including NATIVE, COCO64_GEN, COCO64_156M.
- `train_coco64.py`, training entry that consumes a config.
- `conv_pcn_layer.py`, convolutional PCN layer, supports stride for bidirectional downsampling.
- `dense_pcn_layer.py`, <one accurate line>.
- `transformer_pcn_layer.py`, <one accurate line>.

## Experiment drivers
The `run_*.py` scripts each drive one experiment family (capacity probes, coupling, BPonF baselines, COCO gate and dissociation, E1 baselines). Name the few that are current in `docs/STATE.md`, do not catalog all of them here.

## Downsampling paths (invariant-critical)
- NATIVE uses maxpool and must stay byte-identical.
- COCO64_GEN uses a stride-2 bidirectional conv (conv2d down, transpose-conv up, shared weights) so generative drive flows top-down.

## Why the invariants exist
- Byte-identical NATIVE is the control that every gated comparison depends on.
- The 28.7 GiB size is intentional for the capacity ladder, so OOM means a bug, not a reason to shrink.
```

- [ ] **Step 3: Verify no unfilled placeholders remain**

Run: `grep -n "<" docs/ARCHITECTURE.md || echo "no placeholders"`
Expected: prints `no placeholders`.

- [ ] **Step 4: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "mapped the core modules and downsampling paths in docs/ARCHITECTURE.md"
```

---

### Task 3: docs/STATE.md, the current picture

**Files:**
- Create: `docs/STATE.md`

**Interfaces:**
- Consumes: architecture from Task 2.
- Produces: the note the session protocol reads first.

- [ ] **Step 1: Derive current state from recent signals**

Run: `git log --oneline -10; sed -n '1,15p' CAPACITY.md`
The most recent commit and CAPACITY.md describe where the work is. Summarize into the fields below, keeping it to what is true today.

- [ ] **Step 2: Write docs/STATE.md**

```markdown
# State (updated 2026-07-09)

## Current hypothesis
<one or two lines, e.g. whether the coupling failure persists up the capacity ladder, and where the latest evidence points>

## In flight
- <the thing most recently worked on, from the top commit>

## Next steps
- <the immediate next action>

## Open questions
- <what is unresolved>

## How to update this file
Overwrite the fields above as work moves. Append the detailed run records to `docs/experiments/LOG.md` instead of here.
```

- [ ] **Step 3: Verify**

Run: `grep -q "Current hypothesis" docs/STATE.md && grep -n "<" docs/STATE.md || echo "filled"`
Expected: prints `filled` with no `<` lines remaining.

- [ ] **Step 4: Commit**

```bash
git add docs/STATE.md
git commit -m "seeded docs/STATE.md with the current hypothesis and next steps"
```

---

### Task 4: docs/experiments/LOG.md, the append-only lab notebook

**Files:**
- Create: `docs/experiments/LOG.md`

**Interfaces:**
- Consumes: nothing.
- Produces: the log the session protocol appends to.

- [ ] **Step 1: Gather recent experiments to back-fill**

Run: `git log --oneline -20`
The commit messages are already one-line experiment takeaways. Pick the last 5 to 8 that describe an experiment or outcome and turn each into a log entry.

- [ ] **Step 2: Write docs/experiments/LOG.md**

```markdown
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

### 2026-07-07 strided-conv downsampling ships but text to image still blobs
- config or command, COCO64_GEN with strided-conv downsampling
- result, downsampling trains stably and ships clean, text to image still blobs under the standard relaxation
- takeaway, invertible downsampling is necessary but not sufficient, drive-balance and non-generative training remain the obstacles

<add 4 to 7 more entries from Step 1, newest first, same format>
```

- [ ] **Step 3: Verify**

Run: `grep -c "^### " docs/experiments/LOG.md`
Expected: 5 or more entries. Also run `grep -n "<add" docs/experiments/LOG.md || echo "no placeholders"` and expect `no placeholders`.

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/LOG.md
git commit -m "back-filled docs/experiments/LOG.md from the recent experiment history"
```

---

### Task 5: Obsidian dashboard note

**Files:**
- Create: `~/Documents/Obsidian Vault/1 - Projects/encoder_encoder_PCN.md`

**Interfaces:**
- Consumes: nothing from the repo at runtime, it only links.

- [ ] **Step 1: Write the dashboard note**

```markdown
---
type: project
status: active
tags: [project, research, pcn]
---
# encoder_encoder_PCN

Research PCN testing whether the cross-modal coupling failure persists up the capacity ladder. Source of truth is the repo, this note is a window.

## Repo
`~/encoder_encoder_PCN` (private git). Memory system lives in `CLAUDE.md` and `docs/`.

## Status
See `docs/STATE.md` in the repo for the live picture. Update the one-liner here when a milestone lands.

- Latest, strided-conv downsampling ships, text to image still blobs, invertible downsampling necessary not sufficient.

## Links
- [[Projects MOC]]
```

- [ ] **Step 2: Verify and sync the vault**

Run: `ls "~/Documents/Obsidian Vault/1 - Projects/encoder_encoder_PCN.md"` then from the vault run `git sync`.
Expected: file exists and the vault pushes.

---

### Task 6: Fresh-session smoke test

**Files:** none, verification only.

- [ ] **Step 1: Prove the memory loads cold**

Open a new Claude Code session in `~/encoder_encoder_PCN` and ask, without any other context, "what are the invariants here and what is the current hypothesis?"
Expected: it names the NATIVE byte-identical rule, the OOM-is-a-bug rule, and the current hypothesis from STATE.md, having read CLAUDE.md and STATE.md on its own.

- [ ] **Step 2: Prove the log protocol works**

In that session ask it to record a trivial dummy result, then confirm it appended to the top of `docs/experiments/LOG.md` and did not edit older entries. Revert the dummy entry after.

- [ ] **Step 3: Final commit if any tweaks were needed**

```bash
git add -A
git commit -m "tuned the memory files after the cold-start smoke test"
```

---

## Self-Review

- Spec coverage. CLAUDE.md (Task 1), STATE.md (Task 3), experiments/LOG.md (Task 4), ARCHITECTURE.md (Task 2), Obsidian dashboard (Task 5), seeding (folded into Tasks 1 to 4), session protocol (Task 1), failure-mode coverage verified by Task 6. SessionStart hook is out of scope per the spec and correctly absent.
- Placeholders. The `<...>` markers in Tasks 2, 3, 4 are explicit fill-from-source instructions with a named command producing the content, and each task has a grep step that fails if a marker survives. This is derivation, not an unfilled placeholder.
- Type consistency. File paths and the three doc names are identical across CLAUDE.md, ARCHITECTURE.md, STATE.md, and the tasks.
