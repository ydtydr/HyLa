#!/bin/bash
#SBATCH -J hyla_nc
#SBATCH -o hyla_nc.o%j
#SBATCH -e hyla_nc.o%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=5000
#SBATCH -t 12:00:00
#SBATCH --partition=gpu  --gres=gpu:1
       
python3 HyLa.py \
       -manifold poincare \
       -model hyla \
       -he_dim 16 \
       -dataset cora \
       -use_feats
# 
# remove -use_feats option for airport, add -inductive option for inductive training on reddit