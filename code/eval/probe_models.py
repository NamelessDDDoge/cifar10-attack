"""Probe which pytorchcv CIFAR-10 models download successfully."""
import sys
from pytorchcv.model_provider import get_model

candidates = [
    "resnext29_32x4d_cifar10",
    "resnext29_16x64d_cifar10",
    "shakeshakeresnet20_2x16d_cifar10",
    "shakeshakeresnet26_2x32d_cifar10",
    "rir_cifar10",
    "ror3_56_cifar10",
    "preresnet56_cifar10",
    "seresnet20_cifar10",
    "nin_cifar10",
    "pyramidnet110_a48_cifar10",
    "resdropresnet20_cifar10",
    "diaresnet20_cifar10",
    "diaresnet56_cifar10",
    "wrn28_10_cifar10",
    "wrn40_8_cifar10",
    "seresnet56_cifar10",
    "diapreresnet20_cifar10",
]

for name in candidates:
    try:
        m = get_model(name, pretrained=True)
        print(f"OK  {name}")
    except Exception as e:
        print(f"FAIL {name}: {e}")
    sys.stdout.flush()
