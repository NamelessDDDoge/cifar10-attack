# -*- coding: utf-8 -*-
"""Validate external holdout models on the 500 project images.

Examples:
  python external_clean_accuracy.py --models vit --json-out ../../results/external_vit_clean.json
  python external_clean_accuracy.py --models chenyaofo --device cuda
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from config import IMAGES_DIR, LABEL_FILE
from external_models import (
    EXTERNAL_MODEL_SPECS, EXTERNAL_POOLS,
    get_external_spec, load_external_model, resolve_external_names,
)


def load_labels():
    labels = {}
    with open(LABEL_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, cls = line.split()[:2]
            labels[name] = int(cls)
    return labels


def load_raw_inputs(device):
    inputs = []
    for i in range(500):
        name = f"{i}.png"
        arr = np.array(Image.open(IMAGES_DIR / name).convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        inputs.append((name, t))
    return inputs


@torch.no_grad()
def predict(model, named_inputs, batch_size):
    preds = {}
    for i in range(0, len(named_inputs), batch_size):
        chunk = named_inputs[i:i + batch_size]
        names = [n for n, _ in chunk]
        batch = torch.cat([t for _, t in chunk], dim=0)
        out = model(batch).argmax(dim=1).cpu().tolist()
        preds.update(dict(zip(names, out)))
    return preds


def evaluate_one(model_name, named_inputs, labels, device, batch_size):
    spec = get_external_spec(model_name)
    model = load_external_model(spec, device=device)
    preds = predict(model, named_inputs, batch_size=batch_size)
    correct = sum(preds[n] == labels[n] for n in labels)
    total = len(labels)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {
        "name": spec.name,
        "family": spec.family,
        "loader": spec.loader,
        "model_id": spec.model_id,
        "source_url": spec.source_url,
        "expected_clean_acc": spec.expected_clean_acc,
        "clean_acc_500": correct / total,
        "n_correct": correct,
        "n_total": total,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="external",
                        help="external|vit|robust|chenyaofo|comma-separated external model names")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--json-out", default=str(HERE.parent.parent / "results" / "external_clean_accuracy.json"))
    args = parser.parse_args()

    device = str(torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu")))
    if "," in args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        model_names = resolve_external_names(args.models)

    labels = load_labels()
    named_inputs = load_raw_inputs(device)
    results = []

    print(f"device={device}")
    print(f"models={len(model_names)}")
    print(f"groups={EXTERNAL_POOLS}")

    for model_name in model_names:
        print(f"\n[{model_name}] loading/evaluating ...", flush=True)
        try:
            result = evaluate_one(model_name, named_inputs, labels, device, args.batch_size)
            print(f"  clean_acc_500={result['clean_acc_500']:.4f} "
                  f"({result['n_correct']}/{result['n_total']})")
        except Exception as exc:
            spec = get_external_spec(model_name)
            result = {
                "name": spec.name,
                "family": spec.family,
                "loader": spec.loader,
                "model_id": spec.model_id,
                "source_url": spec.source_url,
                "expected_clean_acc": spec.expected_clean_acc,
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=5),
            }
            print(f"  ERROR: {exc!r}", flush=True)
        results.append(result)

    summary = {
        "device": device,
        "label_file": str(LABEL_FILE),
        "image_dir": str(IMAGES_DIR),
        "n_models": len(model_names),
        "n_ok": sum(r["status"] == "ok" for r in results),
        "n_error": sum(r["status"] == "error" for r in results),
        "results": results,
        "registry": [spec.__dict__ for spec in EXTERNAL_MODEL_SPECS],
    }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
