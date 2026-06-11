#!/usr/bin/env bash
cd "$(dirname "$0")/.." && /c/Users/admin/anaconda3/envs/causal/python.exe -u code/attack/run_awt.py --out-dir results/adv_awt_mixed_smoke20_r4 --max-images 8 --steps 20 --batch-size 1 --cnn-count 2 --vit-surrogates vit_hf_nateraw,vit_timm_edadaltocg --robust-surrogates robust_engstrom,robust_rade_r18_extra,robust_rebuffi_70_16_cutmix_extra,robust_xcit_s12
