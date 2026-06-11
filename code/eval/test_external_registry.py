# -*- coding: utf-8 -*-
"""Fast checks for the external holdout registry. No weight downloads."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from external_models import EXTERNAL_MODEL_SPECS, EXTERNAL_POOLS
from evaluate import Evaluator
from models import HOLDOUT_MODELS, SURROGATE_MODELS

assert EXTERNAL_POOLS["vit"], "ViT external pool is empty"
assert EXTERNAL_POOLS["robust"], "RobustBench external pool is empty"
assert EXTERNAL_POOLS["chenyaofo"], "chenyaofo external pool is empty"

all_names = [spec.name for spec in EXTERNAL_MODEL_SPECS]
assert len(all_names) == len(set(all_names)), "external model names must be unique"
assert not set(all_names) & set(SURROGATE_MODELS), "external names overlap surrogate names"
assert not set(all_names) & set(HOLDOUT_MODELS), "external names overlap pytorchcv holdout names"

for spec in EXTERNAL_MODEL_SPECS:
    assert spec.loader in {"hf_transformers", "timm_hf", "robustbench", "torch_hub"}
    assert spec.source_url.startswith("https://")
    assert 0.0 < spec.expected_clean_acc <= 1.0
    assert spec.family in EXTERNAL_POOLS

print(f"external registry OK: {len(EXTERNAL_MODEL_SPECS)} models")

ev = Evaluator(pools="external", device="cpu")
assert ev.pools == "external"
print("Evaluator accepts external pool")
