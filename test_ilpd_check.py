"""Check pytorchcv model internals: does features.stage2 exist? Any internal normalization?"""
import torch
from pytorchcv.model_provider import get_model
import sys

names = [
    "pyramidnet164_a270_bn_cifar10",
    "wrn16_10_cifar10",
    "densenet190_k40_bc_cifar10",
    "seresnet110_cifar10",
    "resnext29_16x64d_cifar10",
    "diaresnet56_cifar10",
]

for mname in names:
    m = get_model(mname, pretrained=True).eval()
    has_s2 = hasattr(m, 'features') and hasattr(m.features, 'stage2')
    # Check for any normalization-like first layer
    first_layers = [type(v).__name__ for k, v in list(m.named_modules())[:5]]
    # Check input range sensitivity: all-zeros vs all-0.5
    with torch.no_grad():
        x_zero = torch.zeros(1,3,32,32)
        x_half = torch.full((1,3,32,32), 0.5)
        mean = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
        std  = torch.tensor([0.2023,0.1994,0.2010]).view(1,3,1,1)
        x_norm_half = (x_half - mean) / std
        try:
            out_zero = m(x_zero).argmax(1).item()
            out_half = m(x_half).argmax(1).item()
            out_norm = m(x_norm_half).argmax(1).item()
        except Exception as e:
            out_zero = out_half = out_norm = f"ERR:{e}"
    print(f"{mname}: has_stage2={has_s2}  first5={first_layers[:3]}", flush=True)
    sys.stdout.flush()
