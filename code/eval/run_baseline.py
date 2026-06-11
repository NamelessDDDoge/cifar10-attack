# -*- coding: utf-8 -*-
"""
Run both sanity checks and save artifacts/eval-loop/eval_baseline.json.

1. Clean images vs themselves  -> ASR ~ clean error rate (small), SSIM=1
2. Random noise at eps=8/255   -> weak baseline

Usage:
    conda run -n causal python run_baseline.py [--models surrogate|holdout|all]
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMAGES_DIR, WORKSPACE, EPS

# ── Config ────────────────────────────────────────────────────────────────────
EVAL_SCRIPT = Path(__file__).resolve().parent / "evaluate.py"
PROJECT_ROOT = WORKSPACE.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "eval-loop"
OUTPUT_JSON = ARTIFACT_DIR / "eval_baseline.json"

NUM_IMAGES = 500
PYTHON = sys.executable
print(f"[baseline] Using Python: {PYTHON}")


def make_random_noise_dir():
    """Create a temp dir with random-noise adversarial images."""
    tmp = Path(tempfile.mkdtemp(prefix="eval_noise_"))
    rng = np.random.default_rng(42)
    for i in range(NUM_IMAGES):
        img_path = IMAGES_DIR / f"{i}.png"
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        noise = rng.uniform(-EPS, EPS, arr.shape).astype(np.float32)
        adv = np.clip(arr + noise, 0.0, 1.0)
        adv_img = Image.fromarray((adv * 255).astype(np.uint8))
        adv_img.save(tmp / f"{i}.png")
    print(f"[noise] Saved {NUM_IMAGES} noisy images to {tmp}")
    return tmp


def run_evaluate(adv_dir, models, label="?"):
    """Run evaluate.py and return parsed JSON results."""
    tmp_json = Path(tempfile.mktemp(suffix=".json"))
    cmd = [
        PYTHON, str(EVAL_SCRIPT),
        "--adv-dir", str(adv_dir),
        "--models", models,
        "--json-out", str(tmp_json),
    ]
    print(f"\n[{label}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[{label}] stdout:\n{result.stdout}")
    if result.stderr:
        print(f"[{label}] stderr:\n{result.stderr}")
    if result.returncode != 0:
        print(f"[{label}] ERROR: exit code {result.returncode}")
        return None

    with open(tmp_json) as f:
        data = json.load(f)
    tmp_json.unlink(missing_ok=True)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=["surrogate", "holdout", "all"], default="all")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Baseline 1: clean images ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("SANITY CHECK 1: Clean images (expect ASR ~ clean error, SSIM=1)")
    print("="*60)
    clean_results = run_evaluate(IMAGES_DIR, args.models, label="clean")

    # ── Baseline 2: random noise ──────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"SANITY CHECK 2: Random noise eps={EPS:.5f} (weak baseline)")
    print("="*60)
    noise_dir = make_random_noise_dir()
    noise_results = run_evaluate(noise_dir, args.models, label="random_noise")
    shutil.rmtree(noise_dir, ignore_errors=True)

    # ── Assemble output ───────────────────────────────────────────────────────
    output = {
        "clean": clean_results,
        "random_noise": noise_results,
        "surrogate_models": clean_results.get("surrogate_models") if clean_results else None,
        "holdout_models": clean_results.get("holdout_models") if clean_results else None,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[baseline] Written to {OUTPUT_JSON}")

    # ── Print summary ─────────────────────────────────────────────────────────
    for tag, res in [("clean", clean_results), ("random_noise", noise_results)]:
        if res is None:
            print(f"[{tag}] FAILED")
            continue
        for pool in ("surrogate", "holdout"):
            if pool in res:
                r = res[pool]
                print(f"[{tag}/{pool}] ASR={r['pool_asr']:.4f}  "
                      f"SSIM={r['pool_ssim']:.4f}  Score={r['pool_score']:.4f}")


if __name__ == "__main__":
    main()
