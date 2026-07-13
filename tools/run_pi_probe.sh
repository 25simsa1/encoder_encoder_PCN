#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:L4:1
#SBATCH -x n7
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 01:30:00
#SBATCH -J p8_piprobe
#SBATCH -o piprobe_%j.log
export PATH=$HOME/tf-env/bin:$PATH
export LD_LIBRARY_PATH=$(echo $HOME/tf-env/lib/python3.13/site-packages/nvidia/*/lib | tr ' ' ':')
cd $HOME/encoder_encoder_PCN
export PYTHONPATH=$HOME/encoder_encoder_PCN:${PYTHONPATH:-}
set -e
# gate first: the precision defaults must be byte-identical
python3 tools/rewrite_gate.py --config coco64 --relaxed --relax-steps 5 --weight-steps 2 --save pi_cur.npz
python3 tools/gate_compare.py docs/superpowers/gate_ref_coco64.npz pi_cur.npz 1e-4
# read-only probe on the recon best: generative precision schedule, boost off
python3 tools/latent_source_diag.py --ckpt ckpt_gen_best --gamma 0 --pi-bu 0.0 --out latent_source_pi0.png
python3 tools/latent_source_diag.py --ckpt ckpt_gen_best --gamma 0 --pi-bu 0.1 --out latent_source_pi01.png
