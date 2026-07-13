#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:L4:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -J p8_ebm
#SBATCH -o ebm_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
# PC-native EBM by noisy relaxation (contrastive divergence). T0 and ckpt overridable via env.
python3 train_coco64.py --config coco64_gen --train-mode ebm --noise-temp "${T0:-100}" --weight-norm --gen-lr 3e-4 --gen-every 4 --pairs 2000 --epochs 10 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --energy-every 50 --ckpt "${EBM_CKPT:-ckpt_ebm}" --resume
