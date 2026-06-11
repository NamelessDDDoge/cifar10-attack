#!/usr/bin/env bash
# Evaluate adv_ilpd results. Writes eval_ilpd.json.
# WSL: change /c/ -> /mnt/c/

cd /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/code/eval

/c/Users/admin/miniconda3/envs/causal/python.exe -u \
    evaluate.py \
    --adv-dir  /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/results/adv_ilpd \
    --models   all \
    --batch-size 8 \
    --json-out /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/results/eval_ilpd.json
