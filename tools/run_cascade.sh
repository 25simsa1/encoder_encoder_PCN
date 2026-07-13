#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:L4:1
#SBATCH -x n7
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -J p8_cascade
#SBATCH -o cascade_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
# cascade-consistency: per-edge alignment of the top-down cascade to the recon targets
python3 train_coco64.py --config coco64_gen --train-mode cascade \
  --free-state-lr 0.25 --gen-relax-k1 45 --gen-lr "${GLR:-1e-4}" --gen-every 4 --weight-norm \
  --pairs "${PAIRS:-2000}" --epochs "${EPOCHS:-12}" \
  --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 \
  --energy-every "${EEVERY:-50}" --ckpt "${CKPT:-ckpt_cascade}" --resume
