#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:L4:1
#SBATCH -x n7
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 01:30:00
#SBATCH -J p8_piprobe3
#SBATCH -o piprobe3_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
set -e
# the true PC equilibrium at an adequate rate: plain precisions, and a middle pi_bu
python3 tools/latent_source_diag.py --ckpt ckpt_gen_best --gamma 0 --pi-bu 1.0 --decode-state-lr 0.2 --out latent_source_plain_slr02.png
python3 tools/latent_source_diag.py --ckpt ckpt_gen_best --gamma 0 --pi-bu 0.3 --decode-state-lr 0.2 --out latent_source_pib03_slr02.png
