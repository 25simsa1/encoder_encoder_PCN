#!/usr/bin/env bash
# clusterrun.sh — sync local files to the Colby cluster repo and run a command
# on a GPU node SYNCHRONOUSLY via srun (blocks until done, streams output back,
# no client-side polling). Built for the Phase 2 execution-rewrite tests.
#
# Usage:
#   tools/clusterrun.sh --name gate --gpu H200 --mem 64G --cpus 4 --time 00:30:00 \
#       --sync "tools/rewrite_gate.py encoder_encoder_pcn.py" \
#       --run  "python3 tools/rewrite_gate.py --steps 2 --save golden_baseline.npz"
#
# --sync : space-separated repo-relative paths to copy Mac -> cluster (optional)
# --run  : the command to run on the GPU node (env is set up for you: tf-env on
#          PATH, LD_LIBRARY_PATH over the nvidia libs, cwd = repo root)
# Prints the job output. Exits nonzero if srun fails. Run it from anywhere; it
# cd's to the git repo root locally for the sync.
set -euo pipefail

CLUSTER=slsang29@hpc.colby.edu
REPO=encoder_encoder_PCN
GPU=H200; MEM=64G; CPUS=4; TIME=00:30:00; NAME=crun; SYNC=""; RUN=""; FETCH=""
while [[ $# -gt 0 ]]; do case "$1" in
  --gpu)   GPU=$2;   shift 2;;
  --mem)   MEM=$2;   shift 2;;
  --cpus)  CPUS=$2;  shift 2;;
  --time)  TIME=$2;  shift 2;;
  --name)  NAME=$2;  shift 2;;
  --sync)  SYNC=$2;  shift 2;;
  --run)   RUN=$2;   shift 2;;
  --fetch) FETCH=$2; shift 2;;
  *) echo "clusterrun: unknown arg $1" >&2; exit 2;;
esac; done
[[ -z "$RUN" ]] && { echo "clusterrun: --run is required" >&2; exit 2; }

SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 $CLUSTER"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"

if [[ -n "$SYNC" ]]; then
  echo "clusterrun: syncing -> cluster: $SYNC" >&2
  # shellcheck disable=SC2086
  tar -czf - $SYNC | $SSH "cd ~/$REPO && tar -xzf -"
fi

echo "clusterrun: srun on $GPU (mem=$MEM cpus=$CPUS time=$TIME name=$NAME)" >&2
$SSH bash -s <<EOF
set -e
cd \$HOME/$REPO
srun -p gpu --gres=gpu:$GPU:1 -c $CPUS --mem=$MEM -t $TIME -J $NAME \
  bash -lc 'export PATH=\$HOME/tf-env/bin:\$PATH; export LD_LIBRARY_PATH=\$(echo \$HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr " " ":"); cd \$HOME/$REPO; $RUN'
EOF

if [[ -n "$FETCH" ]]; then
  echo "clusterrun: fetching <- cluster: $FETCH" >&2
  # shellcheck disable=SC2086
  $SSH "cd ~/$REPO && tar -czf - $FETCH" | tar -xzf -
fi
