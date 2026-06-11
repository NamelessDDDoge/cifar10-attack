"""List available pytorchcv CIFAR-10 models."""
from pytorchcv.model_provider import _models
cifar_models = [k for k in sorted(_models.keys()) if 'cifar10' in k]
print(f"Total CIFAR-10 models: {len(cifar_models)}")
for m in cifar_models:
    print(m)
