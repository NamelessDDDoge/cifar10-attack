# -*- coding: utf-8 -*-
"""External CIFAR-10 holdout model registry and loaders.

These models are intentionally outside the pytorchcv surrogate/holdout zoo.
They are used to sanity-check transfer against different architectures and
training recipes; attack code must not use them for gradients or tuning.
"""
from dataclasses import dataclass
from typing import Callable, Iterable
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@dataclass(frozen=True)
class ExternalModelSpec:
    name: str
    family: str
    loader: str
    model_id: str
    source_url: str
    expected_clean_acc: float
    input_size: int = 32
    notes: str = ""


VIT_SPECS = [
    ExternalModelSpec(
        name="vit_hf_aaraki",
        family="vit",
        loader="hf_transformers",
        model_id="aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        source_url="https://huggingface.co/aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        expected_clean_acc=0.9788,
        input_size=224,
        notes="ViT-B/16, ImageNet-21k pretrain, CIFAR-10 finetune.",
    ),
    ExternalModelSpec(
        name="vit_hf_nateraw",
        family="vit",
        loader="hf_transformers",
        model_id="nateraw/vit-base-patch16-224-cifar10",
        source_url="https://huggingface.co/nateraw/vit-base-patch16-224-cifar10",
        expected_clean_acc=0.95,
        input_size=224,
        notes="ViT-B/16 CIFAR-10 finetune; model card has no numeric eval table.",
    ),
    ExternalModelSpec(
        name="vit_timm_edadaltocg",
        family="vit",
        loader="timm_hf",
        model_id="hf_hub:edadaltocg/vit_base_patch16_224_in21k_ft_cifar10",
        source_url="https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar10",
        expected_clean_acc=0.9896,
        input_size=224,
        notes="timm ViT-B/16, CIFAR-10 finetune.",
    ),
]


ROBUST_SPECS = [
    ExternalModelSpec(
        name="robust_xcit_s12",
        family="robust",
        loader="robustbench",
        model_id="Debenedetti2022Light_XCiT-S12",
        source_url="https://github.com/RobustBench/robustbench/tree/master/model_info/cifar10/Linf",
        expected_clean_acc=0.88,
        notes="RobustBench CIFAR-10 Linf XCiT-S12.",
    ),
    ExternalModelSpec(
        name="robust_xcit_m12",
        family="robust",
        loader="robustbench",
        model_id="Debenedetti2022Light_XCiT-M12",
        source_url="https://github.com/RobustBench/robustbench/tree/master/model_info/cifar10/Linf",
        expected_clean_acc=0.88,
        notes="RobustBench CIFAR-10 Linf XCiT-M12.",
    ),
    ExternalModelSpec(
        name="robust_rebuffi_70_16_cutmix_extra",
        family="robust",
        loader="robustbench",
        model_id="Rebuffi2021Fixing_70_16_cutmix_extra",
        source_url="https://github.com/RobustBench/robustbench/tree/master/model_info/cifar10/Linf",
        expected_clean_acc=0.90,
        notes="RobustBench CIFAR-10 Linf WRN 70-16, CutMix + extra data.",
    ),
    ExternalModelSpec(
        name="robust_rade_r18_extra",
        family="robust",
        loader="robustbench",
        model_id="Rade2021Helper_R18_extra",
        source_url="https://github.com/RobustBench/robustbench/tree/master/model_info/cifar10/Linf",
        expected_clean_acc=0.88,
        notes="RobustBench CIFAR-10 Linf ResNet-18 style robust model.",
    ),
    ExternalModelSpec(
        name="robust_sehwag_resnest152",
        family="robust",
        loader="robustbench",
        model_id="Sehwag2021Proxy_ResNest152",
        source_url="https://github.com/RobustBench/robustbench/tree/master/model_info/cifar10/Linf",
        expected_clean_acc=0.88,
        notes="RobustBench CIFAR-10 Linf ResNeSt-152 proxy model.",
    ),
    ExternalModelSpec(
        name="robust_engstrom",
        family="robust",
        loader="robustbench",
        model_id="Engstrom2019Robustness",
        source_url="https://github.com/RobustBench/robustbench/tree/master/model_info/cifar10/Linf",
        expected_clean_acc=0.86,
        notes="Classic adversarially trained CIFAR-10 Linf model.",
    ),
]


CHENYA0FO_MODEL_IDS = [
    ("chenyaofo_resnet20", "cifar10_resnet20", 0.9260),
    ("chenyaofo_resnet32", "cifar10_resnet32", 0.9353),
    ("chenyaofo_resnet44", "cifar10_resnet44", 0.9401),
    ("chenyaofo_resnet56", "cifar10_resnet56", 0.9437),
    ("chenyaofo_vgg11_bn", "cifar10_vgg11_bn", 0.9279),
    ("chenyaofo_vgg13_bn", "cifar10_vgg13_bn", 0.9400),
    ("chenyaofo_vgg16_bn", "cifar10_vgg16_bn", 0.9416),
    ("chenyaofo_vgg19_bn", "cifar10_vgg19_bn", 0.9391),
    ("chenyaofo_mobilenetv2_x0_5", "cifar10_mobilenetv2_x0_5", 0.9288),
    ("chenyaofo_mobilenetv2_x0_75", "cifar10_mobilenetv2_x0_75", 0.9372),
    ("chenyaofo_mobilenetv2_x1_0", "cifar10_mobilenetv2_x1_0", 0.9379),
    ("chenyaofo_mobilenetv2_x1_4", "cifar10_mobilenetv2_x1_4", 0.9422),
    ("chenyaofo_shufflenetv2_x0_5", "cifar10_shufflenetv2_x0_5", 0.9013),
    ("chenyaofo_shufflenetv2_x1_0", "cifar10_shufflenetv2_x1_0", 0.9298),
    ("chenyaofo_shufflenetv2_x1_5", "cifar10_shufflenetv2_x1_5", 0.9355),
    ("chenyaofo_shufflenetv2_x2_0", "cifar10_shufflenetv2_x2_0", 0.9381),
    ("chenyaofo_repvgg_a0", "cifar10_repvgg_a0", 0.9439),
    ("chenyaofo_repvgg_a1", "cifar10_repvgg_a1", 0.9489),
    ("chenyaofo_repvgg_a2", "cifar10_repvgg_a2", 0.9498),
]

CHENYA0FO_SPECS = [
    ExternalModelSpec(
        name=name,
        family="chenyaofo",
        loader="torch_hub",
        model_id=model_id,
        source_url="https://github.com/chenyaofo/pytorch-cifar-models",
        expected_clean_acc=acc,
        notes="chenyaofo/pytorch-cifar-models CIFAR-10 pretrained checkpoint.",
    )
    for name, model_id, acc in CHENYA0FO_MODEL_IDS
]


EXTERNAL_MODEL_SPECS = VIT_SPECS + ROBUST_SPECS + CHENYA0FO_SPECS
EXTERNAL_POOLS = {
    "vit": [s.name for s in VIT_SPECS],
    "robust": [s.name for s in ROBUST_SPECS],
    "chenyaofo": [s.name for s in CHENYA0FO_SPECS],
}


def get_external_spec(name: str) -> ExternalModelSpec:
    for spec in EXTERNAL_MODEL_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown external model: {name}")


class NormalizedTorchModel(nn.Module):
    def __init__(self, model: nn.Module, mean: Iterable[float], std: Iterable[float], input_size: int = 32):
        super().__init__()
        self.model = model
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(list(mean), dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(list(std), dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != x.shape[-1]:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return self.model((x - mean) / std)


class HFTransformersImageModel(nn.Module):
    """Differentiable wrapper for HuggingFace ViT image-classification models."""

    def __init__(self, model: nn.Module, input_size: int = 224):
        super().__init__()
        self.model = model
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != x.shape[-1]:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return self.model(pixel_values=(x - mean) / std).logits


def load_external_model(spec_or_name: ExternalModelSpec | str, device: str | torch.device = "cpu") -> nn.Module:
    spec = get_external_spec(spec_or_name) if isinstance(spec_or_name, str) else spec_or_name
    device = torch.device(device)

    if spec.loader == "hf_transformers":
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(spec.model_id)
        wrapped = HFTransformersImageModel(model, input_size=spec.input_size)
    elif spec.loader == "timm_hf":
        import timm
        try:
            model = timm.create_model(spec.model_id, pretrained=True)
        except RuntimeError:
            repo_id = spec.model_id.removeprefix("hf_hub:")
            weights = hf_hub_download(repo_id, "pytorch_model.bin")
            model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10)
            state = torch.load(weights, map_location="cpu")
            model.load_state_dict(state)
        cfg = getattr(model, "pretrained_cfg", {}) or {}
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))
        if "vit_base_patch16_224_in21k_ft_cifar10" in spec.model_id:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        wrapped = NormalizedTorchModel(model, mean=mean, std=std, input_size=spec.input_size)
    elif spec.loader == "robustbench":
        if "autoattack" not in sys.modules:
            stub = types.ModuleType("autoattack")
            stub.__path__ = []
            class AutoAttack:  # pragma: no cover - only satisfies robustbench import side effect.
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("AutoAttack is not available in this eval environment")
            stub.AutoAttack = AutoAttack
            sys.modules["autoattack"] = stub
            state_stub = types.ModuleType("autoattack.state")
            class EvaluationState:  # pragma: no cover
                pass
            state_stub.EvaluationState = EvaluationState
            sys.modules["autoattack.state"] = state_stub
        from robustbench.utils import load_model
        model = load_model(model_name=spec.model_id, dataset="cifar10", threat_model="Linf")
        wrapped = model
    elif spec.loader == "torch_hub":
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", spec.model_id,
            pretrained=True, trust_repo=True)
        wrapped = NormalizedTorchModel(
            model, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), input_size=32)
    else:
        raise ValueError(f"unsupported loader {spec.loader!r}")

    return wrapped.to(device).eval()


def resolve_external_names(selection: str) -> list[str]:
    if selection == "external":
        return [s.name for s in EXTERNAL_MODEL_SPECS]
    if selection in EXTERNAL_POOLS:
        return list(EXTERNAL_POOLS[selection])
    if selection in {s.name for s in EXTERNAL_MODEL_SPECS}:
        return [selection]
    raise KeyError(f"unknown external model selection: {selection}")
