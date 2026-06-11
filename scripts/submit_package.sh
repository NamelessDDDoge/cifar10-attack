#!/usr/bin/env bash
# Full validation + packaging: checks 500 32x32 RGB PNGs, packs submission.zip,
# copies to artifacts/submission/, writes SUBMISSION_REPORT.md.
# Requires adv_images/ to be populated first (run eval_all.sh to pick winner).
# WSL: change /c/ -> /mnt/c/

/c/Users/admin/miniconda3/envs/causal/python.exe -u \
    /c/文件/ME/AISCI/Dr.Researcher/projects/aisafety-cifar10-attack/package_submission.py
