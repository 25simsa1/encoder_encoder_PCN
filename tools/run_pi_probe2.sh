#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:L4:1
#SBATCH -x n7
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 01:30:00
#SBATCH -J p8_piprobe2
#SBATCH -o piprobe2_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
set -e
# read-only probe with an ADEQUATE decode relaxation rate (the as-built 1e-4 was rate-starved)
python3 tools/latent_source_diag.py --ckpt ckpt_gen_best --gamma 0 --pi-bu 0.0 --decode-state-lr 0.2 --out latent_source_pi_slr02.png
python3 tools/latent_source_diag.py --ckpt ckpt_gen_best --gamma 0 --pi-bu 0.0 --decode-state-lr 0.05 --out latent_source_pi_slr005.png
