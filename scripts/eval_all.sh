#!/usr/bin/env bash
# Compare all completed attacks, pick winner by holdout_score,
# copy winner images to adv_images/, write metrics.json and results.json.
# WSL: change /c/ -> /mnt/c/

/c/Users/admin/miniconda3/envs/causal/python.exe -u \
    /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/workspace/code/attack/run_eval_all.py
