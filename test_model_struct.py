from pytorchcv.model_provider import get_model
import torch

for mname in ["seresnet110_cifar10", "wrn16_10_cifar10"]:
    m = get_model(mname, pretrained=True)
    print(f"\n=== {mname} ===")
    for name, mod in list(m.named_modules())[:10]:
        print(f"  {name or 'root'}: {type(mod).__name__}")
    # Check if first conv has specific normalization
    print(f"  in_channels check: ", end="")
    for name, mod in m.named_modules():
        if hasattr(mod, 'running_mean') and 'bn' in name.lower() or 'norm' in name.lower():
            print(f"BN layer: {name}")
            break
