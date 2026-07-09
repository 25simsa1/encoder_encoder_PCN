# Design, project memory system for encoder_encoder_PCN

Date 2026-07-09
Status approved, ready for implementation plan

## Problem

Claude Code loses project-specific instructions and code understanding for encoder_encoder_PCN across four failure modes.

1. Fresh each session. A new session starts cold with no knowledge of the project's conventions, architecture, or prior work.
2. Ignores stated rules. Even after being told a rule, it drifts off it later in the same session.
3. Loses experiment state. It forgets which configs and runs were tried, the results, and the current hypothesis.
4. Mid-session drift. Long sessions lose earlier context as the window fills.

Root cause for the first two is that the repo has no `CLAUDE.md`, the one file Claude Code loads automatically every session. Obsidian alone cannot fix this because Claude does not read the vault automatically.

## Chosen approach

Repo-native knowledge with Obsidian as a browsing window (Approach A of three considered). The always-loaded contract and the living knowledge both live in the git repo so they are versioned with the code and can never drift from it. Obsidian holds a dashboard note that links to the repo for cross-project overview.

Approaches B (Obsidian-primary) and C (split) were rejected because experiment state is tightly coupled to specific configs and files, so keeping it separate from the code invites drift and leans on Claude reliably reading an external vault each session.

## Components

### 1. `CLAUDE.md` at repo root
Auto-loaded every session. Deliberately short so it survives context compaction. Contents.
- One-line project description.
- Hard invariants as a numbered list, seeded from commit history (NATIVE stays byte-identical, OOM is expected so fix bugs not memory, needs a GPU of at least 40GB, plus others found during seeding).
- Key conventions and the config names (NATIVE, COCO64_GEN, COCO64_156M).
- A short architecture orientation that points to `docs/ARCHITECTURE.md`.
- The session protocol (below) written as standing instructions.

### 2. `docs/STATE.md`
The current picture. Current hypothesis, work in flight, next steps, open questions. Updated in place as work moves. This is what a fresh session reads to answer "where are we."

### 3. `docs/experiments/LOG.md`
Append-only lab notebook, dated entries with newest on top. Each entry records the config or command, the result, and a one-line takeaway. Claude appends here after each run. This addresses lost experiment state.

### 4. `docs/ARCHITECTURE.md`
The code map. What the core modules do, how the configs differ, and the invariants explained in prose so the reasons behind them survive.

### 5. Obsidian dashboard note
`1 - Projects/encoder_encoder_PCN.md` in the vault. Links to the repo and carries a short status readable at a glance beside the other projects. It is the window, not the source of truth.

## Session protocol (lives in CLAUDE.md)

- At session start, read `docs/STATE.md` and the top of `docs/experiments/LOG.md` before acting.
- After a work chunk or a run, append to `LOG.md` and update `STATE.md`. Claude owns these updates (user chose auto-update); the user reviews them in the commit diff.

## Seeding

The docs are populated on day one from existing artifacts so they do not start blank. Sources are the git commit history, `CAPACITY.md`, the `analysis_*.md` files, and the Claude memory entry for the project.

## Failure-mode coverage

- Fresh each session. Solved by the auto-loaded CLAUDE.md plus the read-STATE protocol.
- Ignores stated rules. Solved by short invariants that survive compaction.
- Loses experiment state. Solved by the append-only LOG.
- Mid-session drift. Mitigated, not eliminated, by short invariants plus the end-of-chunk log step that re-grounds the session.

## Out of scope for now

- SessionStart hook that injects STATE.md automatically. Optional future strengthening of the cold-start and drift coverage. Left out of the initial build.
- Any change to Obsidian being only a viewer, not a source of truth.

## Success criteria

- A new session opened in the repo correctly states the invariants and current hypothesis without being re-told.
- After a run, the experiment log and STATE reflect it.
- The user can read project status from the Obsidian dashboard note.
