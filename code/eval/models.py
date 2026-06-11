"""Model pools for the CIFAR-10 adversarial attack competition.

All 20 models come from the accuracy-screened TOP20 list (pytorchcv, ranked
by accuracy on the competition images). They are split into two DISJOINT pools:

  - SURROGATE_MODELS (6): attack scripts compute gradients on these.
    Single source of truth: config.SURROGATE_NAMES (one per architecture
    family for gradient diversity).
  - HOLDOUT_MODELS (14): the remaining TOP20 models, NEVER touched by any
    attack script. Their ASR is the unbiased proxy for the hidden judges.
"""
import sys
from pathlib import Path

# config.py lives one level up (workspace/code/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SURROGATE_NAMES

# Accuracy-screened TOP20 (master list, do not reorder/remove).
TOP20_ALL = [
    "pyramidnet164_a270_bn_cifar10",
    "pyramidnet236_a220_bn_cifar10",
    "seresnet110_cifar10",
    "pyramidnet110_a270_cifar10",
    "pyramidnet272_a200_bn_cifar10",
    "resnext29_16x64d_cifar10",
    "wrn16_10_cifar10",
    "densenet190_k40_bc_cifar10",
    "densenet250_k24_bc_cifar10",
    "pyramidnet110_a84_cifar10",
    "wrn20_10_1bit_cifar10",
    "wrn28_10_cifar10",
    "resnet272bn_cifar10",
    "diaresnet56_cifar10",
    "pyramidnet200_a240_bn_cifar10",
    "resnext272_1x64d_cifar10",
    "seresnet164bn_cifar10",
    "wrn20_10_32bit_cifar10",
    "wrn40_8_cifar10",
    "preresnet1001_cifar10",
]

SURROGATE_MODELS = list(SURROGATE_NAMES)

# Guard: surrogates must all come from the screened TOP20.
_unknown = set(SURROGATE_MODELS) - set(TOP20_ALL)
assert not _unknown, f"surrogates not in TOP20: {_unknown}"

# Holdout = TOP20 minus surrogates. Any overlap leaks attack knowledge into eval.
HOLDOUT_MODELS = [m for m in TOP20_ALL if m not in set(SURROGATE_MODELS)]
assert not set(SURROGATE_MODELS) & set(HOLDOUT_MODELS)
assert len(SURROGATE_MODELS) + len(HOLDOUT_MODELS) == len(TOP20_ALL)

# Back-compat alias (old scripts imported TOP20_MODELS). Points to the clean
# holdout pool so any stale import cannot silently re-introduce the leak.
TOP20_MODELS = HOLDOUT_MODELS
