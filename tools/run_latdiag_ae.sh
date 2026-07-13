#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:L4:1
#SBATCH -x n7
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 01:00:00
#SBATCH -J p8_latae
#SBATCH -o latae_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
python3 tools/latent_source_diag.py --ckpt ckpt_chl_ae --weight-norm --gamma 1.0 --out latent_source_ae_g1.png
python3 tools/latent_source_diag.py --ckpt ckpt_chl_ae --weight-norm --gamma 0.0 --out latent_source_ae_g0.png
