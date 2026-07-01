#!/bin/bash
# E1_POD_RUNBOOK.sh -- ordered A100 execution for the venue-gate experiments.
# E1 = BP achievability ladder (run_E1_bp_clip_baseline.py, new)
# E2 = multi-seed the 2k PC negative (existing driver)
# E3 = epochs-vs-data controls (existing driver; a=down-control at 2k, b=matched-epochs at 8k/20k)
#
# ASSUMES: repo at /root/encoder_encoder_PCN, COCO train2017 cache at /root/coco_scale
# (imgs_sc_train2017.npy, >=22k pairs, from the curve run), TF2+PIL+matplotlib env that ran fa1e736.
# Run sections sequentially (one GPU). Each block logs to its own file. Estimated wall times inline,
# extrapolated from committed elapsed fields (2k/150ep arm ~15 min, ~0.36 s/joint-step).
#
# COLLISION RULE: run_coupling_scale.py always writes coupling_scale_results_seed${SEED}.json into its
# own dir. E3 reruns at seed 0 would CLOBBER the committed seed-0 file, so every E3/E2 run executes in
# a scratch copy of the script's dir and the JSON is copied back under a distinct name.
# PRE-CHECK (verified in code, run_coupling_scale.py:162): joint_steps = EPOCHS * ceil(N_train/BATCHJ),
# no independent cap, so EPOCHS env fully controls per-example training at every N.

set -e
cd /root/encoder_encoder_PCN && git pull --rebase
mkdir -p /root/runs

run_pc () {  # run_pc <tag> <env...>   -- scratch-dir wrapper for run_coupling_scale.py
  local TAG=$1; shift
  local D=/root/runs/$TAG; mkdir -p $D; cp run_coupling_scale.py $D/
  ( cd $D && env "$@" RUNS1_COCO=train2017 RUNS1_DATA=/root/coco_scale RUNS1_CKPT=/root/runs/$TAG \
      python3 run_coupling_scale.py 2>&1 | tee /root/runs/$TAG.log )
  cp $D/coupling_scale_results_seed*.json ./res_$TAG.json
  echo "== $TAG done -> res_$TAG.json =="
}

# ---------------------------------------------------------------- PHASE 1, the hinge (~30-40 min)
# E1 rung 1: BP at 2k, 3 seeds. Early-stops once the train-fit gate passes.
RUNS1_NTRAIN=2000 RUNS1_NEVAL=1000 E1_SEEDS=0,1,2 \
  python3 run_E1_bp_clip_baseline.py 2>&1 | tee /root/runs/E1_2k.log
git add E1_results.json run_E1_bp_clip_baseline.py && git commit -m "E1 achievability baseline, 2k rung" && git push

# ---------------------------------------------------------------- PHASE 2, cheap controls (~35 min)
# E3a down-control: 2k at 45 then 18 epochs, seed 0, control on. Isolates epochs from data:
# if 2k@45ep reproduces the 8k diversity drop (0.13-0.15), the drop is an epochs artifact, not scale.
run_pc 2k_45ep  RUNS1_NTRAIN=2000 RUNS1_NEVAL=1000 RUNS1_EPOCHS=45 RUNS1_SEED=0 RUNS1_CONTROL=1
run_pc 2k_18ep  RUNS1_NTRAIN=2000 RUNS1_NEVAL=1000 RUNS1_EPOCHS=18 RUNS1_SEED=0 RUNS1_CONTROL=1
git add res_2k_45ep.json res_2k_18ep.json && git commit -m "E3a epochs down-control at 2k" && git push

# ---------------------------------------------------------------- PHASE 3, multi-seed negative (~2.5h)
# E2: 2k at 150 epochs, seeds 0,1,2 on train2017. Seed 0 is included because the committed 2k seed-0
# point (coupling_scale_results_seed0.json) was val2017 with N_have=3000, a different split; a clean
# 3-seed set needs all three on the same train2017 cache.
run_pc 2k_150ep_s0 RUNS1_NTRAIN=2000 RUNS1_NEVAL=1000 RUNS1_EPOCHS=150 RUNS1_SEED=0 RUNS1_CONTROL=1
run_pc 2k_150ep_s1 RUNS1_NTRAIN=2000 RUNS1_NEVAL=1000 RUNS1_EPOCHS=150 RUNS1_SEED=1 RUNS1_CONTROL=1
run_pc 2k_150ep_s2 RUNS1_NTRAIN=2000 RUNS1_NEVAL=1000 RUNS1_EPOCHS=150 RUNS1_SEED=2 RUNS1_CONTROL=1
git add res_2k_150ep_s*.json && git commit -m "E2 multi-seed 2k negative, train2017" && git push

# ---------------------------------------------------------------- PHASE 4, E1 ladder up (~1-3h)
RUNS1_NTRAIN=8000 RUNS1_NEVAL=2000 E1_SEEDS=0,1,2 \
  python3 run_E1_bp_clip_baseline.py 2>&1 | tee /root/runs/E1_8k.log
git add E1_results.json && git commit -m "E1 achievability, 8k rung" && git push
RUNS1_NTRAIN=20000 RUNS1_NEVAL=2000 E1_SEEDS=0,1,2 E1_EPOCHS=100 \
  python3 run_E1_bp_clip_baseline.py 2>&1 | tee /root/runs/E1_20k.log
# NOTE: if the TRAIN-FIT gate fails at a rung (train lat_retr < 0.5 at budget), raise E1_EPOCHS and rerun
# that rung before trusting its held-out number. The held-out claim requires the fit gate.
git add E1_results.json && git commit -m "E1 achievability, 20k rung" && git push

# ---------------------------------------------------------------- PHASE 5, matched-epochs (~3h + ~7.5h)
# E3b co-critical: does held-out stay at chance at 8k/20k when per-example training matches 2k (150ep)?
# 8k@150ep: 9450 joint steps/arm, ~57 min/arm, 3 arms ~3h. 20k@150ep: 23550 steps/arm, ~7.5h total.
run_pc 8k_150ep  RUNS1_NTRAIN=8000  RUNS1_NEVAL=2000 RUNS1_EPOCHS=150 RUNS1_SEED=0 RUNS1_CONTROL=1
git add res_8k_150ep.json && git commit -m "E3b matched-epochs at 8k" && git push
run_pc 20k_150ep RUNS1_NTRAIN=20000 RUNS1_NEVAL=2000 RUNS1_EPOCHS=150 RUNS1_SEED=0 RUNS1_CONTROL=1
git add res_20k_150ep.json && git commit -m "E3b matched-epochs at 20k" && git push

# ---------------------------------------------------------------- PHASE 6, conditional ladder extension
# ONLY if no E1 rung crossed the bar by 20k. The cache holds 22k pairs; climbing needs a bigger cache.
# Previously downloaded images are reused (DATA/img persists); only the .npy/.txt need rebuilding.
# mv /root/coco_scale/imgs_sc_train2017.npy /root/coco_scale/imgs_sc_train2017.npy.22k
# mv /root/coco_scale/caps_sc_train2017.txt /root/coco_scale/caps_sc_train2017.txt.22k
# RUNS1_NTRAIN=40000 RUNS1_NEVAL=2000 RUNS1_PAIRS=44000 E1_SEEDS=0 E1_EPOCHS=60 \
#   python3 run_E1_bp_clip_baseline.py 2>&1 | tee /root/runs/E1_40k.log
# RUNS1_NTRAIN=80000 RUNS1_NEVAL=2000 RUNS1_PAIRS=86000 E1_SEEDS=0 E1_EPOCHS=40 \
#   python3 run_E1_bp_clip_baseline.py 2>&1 | tee /root/runs/E1_80k.log
# If still no cross, fire the labeled ceiling (non-from-scratch, branch b vs c):
# RUNS1_NTRAIN=2000 RUNS1_NEVAL=1000 E1_SEEDS=0 E1_ORACLE=1 python3 run_E1_bp_clip_baseline.py \
#   2>&1 | tee /root/runs/E1_oracle.log
# git add E1_results.json && git commit -m "E1 ladder extension / oracle ceiling" && git push
