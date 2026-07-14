#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:HEQ:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 02:00:00
#SBATCH -J p8_untret
#SBATCH -o untret_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
set -e
python3 tools/latent_source_diag.py --ckpt ckpt_untied_best --untied --td-ckpt ckpt_untied_best_td --gamma 0 --pi-bu 0.0 --decode-state-lr 0.25 --out latent_source_unt_pi.png
python3 tools/latent_source_diag.py --ckpt ckpt_untied_best --untied --td-ckpt ckpt_untied_best_td --gamma 0 --pi-bu 1.0 --decode-state-lr 0.2 --out latent_source_unt_plain.png
python3 tools/latent_source_diag.py --ckpt ckpt_untied_best --untied --td-ckpt ckpt_untied_best_td --gamma 1.0 --out latent_source_unt_boost.png
