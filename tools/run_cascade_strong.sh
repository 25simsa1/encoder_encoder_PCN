#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:HEQ:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -J p8_cascS
#SBATCH -o cascS_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
# STRONG-pressure cascade-consistency: calibration at recon-equal rate and frequency (40x the
# gentle cell), the decisive test of the per-edge calibration family
python3 train_coco64.py --config coco64_gen --train-mode cascade \
  --free-state-lr 0.25 --gen-relax-k1 45 --gen-lr 1e-3 --gen-every 1 --weight-norm \
  --pairs 2000 --epochs 10 \
  --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 \
  --energy-every 50 --ckpt ckpt_cascade_s --resume
