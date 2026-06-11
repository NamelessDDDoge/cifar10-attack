#!/usr/bin/env bash
# Evaluate run_strong adv images.
# WSL: change /c/ -> /mnt/c/

cd /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/code/eval

/c/Users/admin/miniconda3/envs/causal/python.exe -u \
    evaluate.py \
    --adv-dir  /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/results/run_strong/adv_images \
    --models   all \
    --json-out /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/results/run_strong/eval.json
