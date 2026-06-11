# -*- coding: utf-8 -*-
"""Quick smoke test: pools disjoint, labels load, Evaluator constructs.
No model downloads, no full eval. Run: conda run -n causal python smoke_test.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models import SURROGATE_MODELS, HOLDOUT_MODELS, TOP20_MODELS, TOP20_ALL

assert len(TOP20_ALL) == 20, len(TOP20_ALL)
assert len(SURROGATE_MODELS) == 6, SURROGATE_MODELS
assert len(HOLDOUT_MODELS) == 14, len(HOLDOUT_MODELS)
assert not set(SURROGATE_MODELS) & set(HOLDOUT_MODELS), "pool overlap!"
assert set(SURROGATE_MODELS) | set(HOLDOUT_MODELS) == set(TOP20_ALL)
assert TOP20_MODELS is HOLDOUT_MODELS
print(f"pools OK: surrogate={len(SURROGATE_MODELS)} holdout={len(HOLDOUT_MODELS)} disjoint")

import evaluate

labels = evaluate.load_labels()
assert len(labels) == 500
assert all(0 <= v <= 9 for v in labels.values())
print(f"labels OK: {len(labels)} entries, sample {list(labels.items())[:3]}")

ev = evaluate.Evaluator(pools="all", device="cpu")
print(f"Evaluator OK: pools={ev.pools} device={ev.device}")

clean = ev._load_clean_images()
assert len(clean) == 500
adv, n_missing = ev._load_adv_images(HERE.parent.parent / "results" / "adv_ilpd", clean)
assert len(adv) == 500
print(f"image loading OK: 500 clean, 500 adv (missing={n_missing})")

# SSIM spot check: clean vs itself = 1.0
from skimage.metrics import structural_similarity as ssim
name, arr, _ = clean[0]
s = ssim(arr, arr, channel_axis=2, data_range=1.0)
assert abs(s - 1.0) < 1e-9
print("SSIM identity check OK (=1.0)")

print("\nALL SMOKE TESTS PASSED")
