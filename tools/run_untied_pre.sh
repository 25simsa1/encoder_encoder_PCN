#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:HEQ:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 01:00:00
#SBATCH -J p8_untpre
#SBATCH -o untpre_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
python3 tools/rewrite_gate.py --config coco64 --relaxed --relax-steps 5 --weight-steps 2 --save unt_cur.npz
echo GATE_2E4_ENVELOPE:
python3 tools/gate_compare.py docs/superpowers/gate_ref_coco64.npz unt_cur.npz 2e-4 || true
echo SMOKE:
python3 train_coco64.py --config coco64_gen --untied --pairs 64 --epochs 2 --lr 1e-3 --weight-decay 3e-2 --state-clip 400 --conv-activation gelu --relax 15 --batch 8 --energy-every 4 --ckpt ckpt_unt_smoke
