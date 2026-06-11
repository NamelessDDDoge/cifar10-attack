#!/usr/bin/env bash
# Pack adv_ilpd images into submission_ilpd.zip (lossless PNG).
# WSL: change /c/ -> /mnt/c/

/c/Users/admin/miniconda3/envs/causal/python.exe -u \
    /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/code/attack/make_submission.py \
    --adv-dir   /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/results/adv_ilpd \
    --label-txt /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/data/label.txt \
    --out-zip   /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/results/submission_ilpd.zip
