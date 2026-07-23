# Results figure redesign (ICLR body) — design spec

## Goal
Replace fig2/fig3 with ONE double-column hero strip (fig2_results.pdf + .png preview,
6.75x2.2in, 8pt, colorblind-safe blue #0173b2 = backprop / orange #de8f05 = PC, no top/right
spines, panel letters) telling the full argument left to right.

## Panels
(a) Matched training fit — per-seed paired dots (train latent retrieval): BP .981/.951/.972,
    PC .997/.998/.999; y in [0.9,1.005]; message "both at ceiling" (fairness proof).
(b) Held-out category transfer (HEADLINE) — mean precision@10 bars w/ 3 seed dots overlaid:
    BP .2146/.2001/.1995, PC .0998/.1080/.1023; dashed labeled base-rate line (.0815 mean);
    annotation "~2.5x base" over BP, "~1.3x" over PC.
(c) Per-category systematicity — dumbbell plot, seed-0 lifts, sorted by BP, log-x, line
    connecting PC dot -> BP dot per category; categories: elephant 7.2/1.1, plane 6.1/2.4,
    cat 5.6/1.4, food 5.0/1.8, train 4.3/1.5, sports 3.8/1.8, bathroom 3.2/2.0, dog 1.8/0.6,
    cow_sheep 1.6/1.3, kitchen 1.3/1.8, person 1.2/0.9, sign 1.0/0.8, bird 0.5/0.8; vertical
    line at 1x labeled "base rate".

## Implementation
One matplotlib script (tools/make_results_figure.py, numbers inline w/ source comments),
outputs docs/paper/figs/fig2_results.{pdf,png}; old fig2/fig3 kept until draft references
updated; PAPER_DRAFT_v2.md Figures section updated. Instance-hit numbers move to a small
table in the draft (BP 18/12/11 vs PC 3/7/2), not a panel.

## Acceptance
Legible at 100% zoom in a 2-col PDF; each panel answers its objection (undertrained? /
noise? / cherry-picked?); no chartjunk; fonts embedded in PDF.
