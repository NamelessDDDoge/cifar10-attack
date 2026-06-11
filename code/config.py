# -*- coding: utf-8 -*-
"""
Shared path configuration for the CIFAR-10 adversarial attack project.
All scripts should import paths from here to avoid hardcoded inconsistencies.
"""
from pathlib import Path

# Project root (workspace/)
WORKSPACE = Path(__file__).resolve().parent.parent

# Data
IMAGES_DIR = WORKSPACE / "data" / "images"
LABEL_FILE = WORKSPACE / "data" / "label.txt"

# Code
EVAL_DIR = WORKSPACE / "code" / "eval"
ATTACK_DIR = WORKSPACE / "code" / "attack"
REPOS_DIR = WORKSPACE / "code" / "repos"

# Results
RESULTS_DIR = WORKSPACE / "results"
ADV_IMAGES_DIR = RESULTS_DIR / "adv_images"

# Cache
CACHE_DIR = WORKSPACE / "cache"

# CIFAR-10 constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Surrogate model pool (used by attack scripts).
# 6 of the accuracy-screened TOP20 models, one per architecture family for
# gradient diversity. The remaining 14 TOP20 models form the holdout pool
# (eval/models.py) and must NEVER be touched by attack code.
SURROGATE_NAMES = [
    "pyramidnet164_a270_bn_cifar10",
    "wrn16_10_cifar10",
    "densenet190_k40_bc_cifar10",
    "seresnet110_cifar10",
    "resnext29_16x64d_cifar10",
    "diaresnet56_cifar10",
]

# Attack hyperparameters
EPS = 8.0 / 255.0
ALPHA = 2.0 / 255.0
STEPS = 10
BATCH_SIZE = 10
