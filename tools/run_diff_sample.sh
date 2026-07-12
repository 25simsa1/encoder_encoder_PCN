#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:L4:1
#SBATCH -x n7
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 00:20:00
#SBATCH -J p8_dsamp
#SBATCH -o dsamp_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
python3 tools/diffusion_sample.py --ckpt ckpt_diff --levels 10 --sigma-min 0.05 --sigma-max 0.8 --k 15 --draws 3 --out diff_samples.png
