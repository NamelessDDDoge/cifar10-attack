# -*- coding: utf-8 -*-
"""
Evaluate all completed attack output directories and collect results.
Writes results per-algo to individual JSON files.
Run after all attacks complete.

Winner selection uses HOLDOUT score only (proxy for hidden judges).
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, EVAL_DIR, WORKSPACE

RESULTS = RESULTS_DIR
EVAL_PY = EVAL_DIR / "evaluate.py"
PYTHON = sys.executable

ALGOS = {
    "pgn":  {"dir": RESULTS / "adv_pgn",  "repo": "https://github.com/Trustworthy-AI-Group/PGN"},
    "sit":  {"dir": RESULTS / "adv_sit",  "repo": "https://github.com/xiaosen-wang/SIT"},
    "awt":  {"dir": RESULTS / "adv_awt",  "repo": "https://github.com/xaddwell/AWT"},
    "bsr":  {"dir": RESULTS / "adv_bsr",  "repo": "https://github.com/Trustworthy-AI-Group/TransferAttack"},
    "ilpd": {"dir": RESULTS / "adv_ilpd", "repo": "https://github.com/qizhangli/ILPD-attack"},
}

ADAPT_NOTES = {
    "pgn":  "Replaced inceptionv3 with pytorchcv CIFAR-10 ensemble (6 surrogates); eps=8/255; removed ImageNet CSV loader; N_SAMPLES=10",
    "sit":  "Passed pytorchcv CIFAR-10 ensemble as model directly; num_block=4 (32x32/8=4); num_copies=10; eps=8/255",
    "awt":  "Overrode load_model() to return pytorchcv CIFAR-10 ensemble; SAM tunes primary model weights only (OOM-safe); eps=8/255",
    "bsr":  "TransferAttack BSR with pytorchcv CIFAR-10 surrogates; num_block adapted for 32x32; eps=8/255",
    "ilpd": "Replaced timm models with pytorchcv pyramidnet164_a270_bn primary + 5 surrogates; hooked features.stage2; eps=8/255; coef=0.1",
}

# Cached eval JSON must have these keys AND the current model pools, else it
# predates a metric/pool change and must be recomputed.
REQUIRED_KEYS = {"surrogate", "holdout", "mean_ssim"}

sys.path.insert(0, str(EVAL_DIR))
from models import SURROGATE_MODELS, HOLDOUT_MODELS


def cache_is_current(data):
    if not REQUIRED_KEYS.issubset(data.keys()):
        return False
    return (data["surrogate"].get("models") == SURROGATE_MODELS
            and data["holdout"].get("models") == HOLDOUT_MODELS)


def eval_algo(name, adv_dir):
    png_count = len(list(adv_dir.glob("*.png")))
    if png_count != 500:
        print(f"[skip] {name}: only {png_count}/500 images in {adv_dir}")
        return None

    out_json = RESULTS / f"eval_{name}.json"
    if out_json.exists():
        with open(out_json) as f:
            data = json.load(f)
        if cache_is_current(data):
            print(f"[cached] {name}: reading {out_json}")
            return data
        print(f"[stale] {name}: {out_json} uses old metric schema or pools, re-evaluating")

    print(f"[eval] Running evaluate.py for {name} ...")
    cmd = [
        PYTHON, str(EVAL_PY),
        "--adv-dir", str(adv_dir),
        "--models", "all",
        "--json-out", str(out_json),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"[error] evaluate.py failed for {name}:\n{result.stderr[-500:]}")
        return None

    with open(out_json) as f:
        return json.load(f)


def main():
    per_algo = {}
    for name, cfg in ALGOS.items():
        data = eval_algo(name, cfg["dir"])
        if data is None:
            continue
        surr = data["surrogate"]
        hold = data["holdout"]
        per_algo[name] = {
            "repo": cfg["repo"],
            "adapted": ADAPT_NOTES[name],
            "surrogate_asr": round(surr["pool_asr"], 4),
            "holdout_asr": round(hold["pool_asr"], 4),
            "ssim": round(data["mean_ssim"], 4),
            "surrogate_score": round(surr["pool_score"], 4),
            "holdout_score": round(hold["pool_score"], 4),
        }

    if not per_algo:
        print("No completed attacks found.")
        return

    # Winner by holdout_score
    winner = max(per_algo, key=lambda n: per_algo[n]["holdout_score"])
    winner_score = per_algo[winner]["holdout_score"]
    print(f"\nWinner: {winner} (holdout_score={winner_score})")

    # Print table
    print(f"\n{'Algo':<8} {'Surr_ASR':>10} {'Hold_ASR':>10} {'SSIM':>8} {'Surr_Score':>12} {'Hold_Score':>12}")
    for n, v in per_algo.items():
        mark = " <-- WINNER" if n == winner else ""
        print(f"{n:<8} {v['surrogate_asr']:>10.4f} {v['holdout_asr']:>10.4f} {v['ssim']:>8.4f} {v['surrogate_score']:>12.4f} {v['holdout_score']:>12.4f}{mark}")

    # Copy winner images to adv_images
    import shutil
    adv_images = RESULTS / "adv_images"
    adv_images.mkdir(exist_ok=True)
    winner_dir = ALGOS[winner]["dir"]
    print(f"\nCopying {winner} images to adv_images ...")
    for png in winner_dir.glob("*.png"):
        shutil.copy2(png, adv_images / png.name)
    print(f"Copied {len(list(winner_dir.glob('*.png')))} images.")

    # Write metrics.json for the winner (consumed by package_submission.py)
    w = per_algo[winner]
    metrics = {
        "algorithm": winner.upper(),
        "ASR": w["holdout_asr"],
        "SSIM": w["ssim"],
        "Score": w["holdout_score"],
        "note": "holdout-pool metrics (proxy for hidden judges); SSIM over all 500 images",
    }
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {RESULTS / 'metrics.json'}")

    # Write results.json
    results = {
        "per_algo": per_algo,
        "winner": winner,
        "winner_holdout_score": winner_score,
    }
    out_results = WORKSPACE.parent / "artifacts" / "attack" / "results.json"
    out_results.parent.mkdir(parents=True, exist_ok=True)
    with open(out_results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_results}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
