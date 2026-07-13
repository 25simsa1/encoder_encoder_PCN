#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:L4:1
#SBATCH -x n7
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 01:30:00
#SBATCH -J p8_cascret
#SBATCH -o cascret_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
set -e
# the schedule the training calibrated for, plus the boost baseline
python3 tools/latent_source_diag.py --ckpt ckpt_cascade --weight-norm --gamma 0 --pi-bu 0.0 --decode-state-lr 0.25 --out latent_source_casc_pi.png
python3 tools/latent_source_diag.py --ckpt ckpt_cascade --weight-norm --gamma 1.0 --out latent_source_casc_boost.png
