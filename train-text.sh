#!/bin/bash
#SBATCH -J hyla_text
#SBATCH -o hyla_text.o%j
#SBATCH -e hyla_text.o%j
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=5000
#SBATCH -t 120:00:00
#SBATCH --partition=gpu  --gres=gpu:1
       
python3 TextHyLa.py \
       -manifold poincare \
       -model laplaNN \
       -dim 30 \
       -HyLa_fdim 500 \
       -order 2 \
       -seed 43 \
       -scale 0.5 \
       -lr_e 0.01 \
       -lr_c  0.0001 \
       -epochs 100 \
       -task text \
       -dataset R8 \
       -metric acc \
       -use_feats

# poincare, laplaNN
# -batchtraining -use_feats
# nc unnormalized features
# sgc -fresh -checkpoint
# feature level 
# R8 50d 1000 2order 0.5scale 0.001lre constant 0.0001lrc 100epoch adam sgc 97.35%test
# R52 50d 1000 2order 0.5scale 0.008lre constant 0.0001lrc 100epoch adam sgc 93.5%test 
# ohsumed 50d 1000 2order 0.5scale 0.001lre constant 0.0001lrc 100epoch adam sgc 64.85%test
# ohsumed 50d 1000 2order 0.1scale 0.001lre constant 0.0001lrc 200epoch adam sgc 65.72%test
# 20ng 30d 500 2order 0.5scale 0.01lre constant 0.0001lrc 100epoch adam sgc 77.71%test (78.92%) though val 87.44% *88.06%)
# 20ng with new datasplit 30d 500 2order 0.5scale 0.01lre constant 0.0001lrc 100epoch adam sgc 84.85%test val 82.67%
# mr 50d 500 2order 0.5scale 0.01lre constant 0.0001lrc 100epoch adam sgc mr 75.49%test

# all levels 
# R8 50d 500 2order 0.5scale 0.1lre constant 0.0001lrc 100epoch adam sgc 96.62%test
# R8 50d 500 2order 0.5scale 0.01lre constant 0.0001lrc 200epoch adam sgc 96.94%test

# R52 50d 500 2order 0.5scale 0.1lre constant 0.0001lrc 100epoch adam sgc 93.46%test 
# R52 50d 500 2order 0.5scale 0.1lre constant 0.0001lrc 200epoch adam sgc 94.04%test

# ohsumed 50d 500 2order 0.5scale 0.1lre constant 0.0001lrc 100epoch adam sgc 66.61%test
# ohsumed 50d 500 2order 0.5scale 0.01lre constant 0.0001lrc 200epoch adam sgc 67.33%test

# mr 30d 500 2order 0.5scale 0.1lre constant 0.0001lrc 100epoch adam sgc mr 75.44%test
# mr 30d 500 2order 0.5scale 0.1lre constant 0.0001lrc 200epoch adam sgc mr 76.17%test

# 20ng 30d 500 2order 0.5scale 0.1lre constant 0.0001lrc 100epoch adam sgc %test


# euclidean, EuclaplaNN
# R8 50d 1000 2order 0.5scale 1.0lre sgd 0.001lrc 100epoch adam sgc 97.21%test feature level 
# R8 50d 1000 2order 0.5scale 0.01lre adam 0.001lrc 100epoch adam sgc 93.51%test feature level 
# R8 50d 1000 2order 0.5scale 2.0lre sgd 0.001lrc 100epoch adam sgc 92.69%test all level 
# R8 50d 1000 2order 0.5scale 0.1lre adam 0.001lrc 100epoch adam sgc 96.53%test all level 
# R8 LR nonemodel 0.01lrc 100epoch adam sgc 93.33%test feature level 

# R52 50d 1000 2order 0.5scale 3.0lre sgd 0.001lrc 100epoch adam sgc 92.17%test feature level 
# R52 50d 1000 2order 0.5scale 0.01lre adam 0.001lrc 100epoch adam sgc 88.75%test feature level
# R52 50d 1000 2order 0.5scale 5.0lre sgd 0.001lrc 100epoch adam sgc 79.91%test all level 
# R52 50d 1000 2order 0.5scale 3.0lre sgd 0.001lrc 200epoch adam sgc 86.29%test all level
# R52 50d 1000 2order 0.5scale 0.1lre adam 0.001lrc 100epoch adam sgc 94.00%test all level 
# R52 LR nonemodel 0.01lrc 100epoch adam sgc 85.63%test feature level 

# ohsumed 50d 1000 2order 0.5scale 1.0lre sgd 0.001lrc 100epoch adam sgc 61.56%test feature level 
# ohsumed 50d 1000 2order 0.5scale 0.01lre adam 0.001lrc 100epoch adam sgc 60.20%test feature level 
# ohsumed 50d 1000 2order 0.5scale 2.0lre sgd 0.001lrc 100epoch adam sgc 38.12%test all level 
# ohsumed 50d 1000 2order 0.5scale 0.1lre adam 0.001lrc 100epoch adam sgc 67.15%test all level 
# ohsumed LR nonemodel 0.01lrc 100epoch adam sgc 56.62%test feature level 

# mr 50d 1000 2order 0.5scale 5.0lre sgd 0.001lrc 100epoch adam sgc 76.03%test feature level 
# mr 50d 1000 2order 0.5scale 0.001lre adam 0.001lrc 100epoch adam sgc 73.55%test feature level 
# mr 50d 1000 2order 0.5scale 0.01lre adam 0.001lrc 100epoch adam sgc 73.10%test all level
# mr LR nonemodel 0.001lrc 100epoch adam sgc 73.04%test feature level 

# 20ng 50d 1000 2order 0.5scale 1.0lre sgd 0.001lrc 100epoch adam sgc 78.58%test feature level
# 20ng 50d 1000 2order 0.5scale 0.01lre adam 0.001lrc 100epoch adam sgc 80.05%test feature level
# 20ng 50d 1000 2order 0.5scale 0.1lre adam 0.001lrc 100epoch adam sgc 5.01%test all level
# 20ng LR nonemodel 0.01lrc 100epoch adam sgc 80.20%test feature level 