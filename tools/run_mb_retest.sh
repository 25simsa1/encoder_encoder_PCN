#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:HEQ:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -J p8_mbret
#SBATCH -o mbret_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
set -e
# multi-branch boost on the BEST decode artifact: all 5 latents inject (rank-500 vs rank-100)
python3 tools/latent_source_diag.py --ckpt ckpt_untied5 --untied --td-ckpt ckpt_untied5_td --gamma 1.0 --rms-match --multi-branch --out latent_source_mb_boost.png
python3 tools/latent_source_diag.py --ckpt ckpt_untied5 --untied --td-ckpt ckpt_untied5_td --gamma 0.5 --rms-match --multi-branch --out latent_source_mb_boost05.png
