#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:L4:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -J p8_hf
#SBATCH -o hf_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
# gamma and ckpt overridable via env (sbatch --export); defaults = the first probe run
python3 train_coco64.py --config coco64_gen --hf-weight "${HF_GAMMA:-1.0}" --weight-norm --train-mode recon --pairs 2000 --epochs 12 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --energy-every 50 --ckpt "${HF_CKPT:-ckpt_hf_g1}" --resume
