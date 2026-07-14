#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:HEQ:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -J p8_untied
#SBATCH -o untied_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
# from-scratch recon with UNTIED top-down weights + the isometry stabilizer on BOTH matrices
# (the unopposed td step norm-inflated off a cliff at ep2 without it)
python3 train_coco64.py --config coco64_gen --untied --isometry "${ETA:-1e-3}" \
  --pairs "${PAIRS:-2000}" --epochs "${EPOCHS:-15}" \
  --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 \
  --energy-every "${EEVERY:-50}" --ckpt "${CKPT:-ckpt_untied}"
