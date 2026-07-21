#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:H200:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -J p4_inf_l10
#SBATCH -o infonce_l10_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
python3 train_coco64.py --pairs 2000 --epochs 15 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --infonce-lambda 1.0 --infonce-tau 0.07 --relax 15 --batch 8 --energy-every 50 --ckpt ckpt_infonce_l10
