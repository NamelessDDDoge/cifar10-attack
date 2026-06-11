#!/usr/bin/env bash
# Stronger run: 50 steps, larger ensemble
# WSL: change /c/ -> /mnt/c/

/c/Users/admin/miniconda3/envs/causal/python.exe -u \
    /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/code/attack/attack.py \
    --images-dir /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/data/images \
    --work-dir   /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/results/run_strong \
    --eps        0.03137255 \
    --steps      50 \
    --step-size  0.00313726 \
    --momentum   1.0 \
    --diversity-prob 0.7 \
    --admix-eta  0.2 \
    --admix-m2   5 \
    --ensemble   resnet50,vgg19,densenet121,mobilenetv2 \
    --seed       42
