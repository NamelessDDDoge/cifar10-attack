"""
Per-model pseudo-label accuracy vs true CIFAR-10 labels.
"""
import numpy as np
import torch
from pathlib import Path
from PIL import Image

IMAGES_DIR = Path(r"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\datasets\images")
TRUE_LABELS_FILE = Path(r"C:\Users\admin\Desktop\AISafety\_vera\work\true_labels.txt")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

ALL_MODELS = [
    ("resnet20_cifar10",                  "surrogate"),
    ("resnet56_cifar10",                  "surrogate"),
    ("wrn16_10_cifar10",                  "surrogate"),
    ("densenet40_k12_cifar10",            "surrogate"),
    ("shakeshakeresnet20_2x16d_cifar10",  "surrogate"),
    ("preresnet56_cifar10",               "holdout"),
    ("seresnet20_cifar10",                "holdout"),
    ("nin_cifar10",                       "holdout"),
    ("pyramidnet110_a48_cifar10",         "holdout"),
    ("ror3_110_cifar10",                  "holdout"),
]

def load_true_labels():
    true_labels = {}
    with open(TRUE_LABELS_FILE) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            name, true_l = parts[0], int(parts[1])
            true_labels[name] = true_l
    return true_labels

def load_images(device):
    mean = torch.tensor(CIFAR10_MEAN).view(3,1,1)
    std  = torch.tensor(CIFAR10_STD).view(3,1,1)
    inputs = []
    names = []
    for i in range(500):
        name = f"{i}.png"
        arr = np.array(Image.open(IMAGES_DIR / name).convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2,0,1))
        inp = ((t - mean) / std).unsqueeze(0).to(device)
        inputs.append(inp)
        names.append(name)
    return names, inputs

@torch.no_grad()
def predict_all(model, inputs, device, batch=64):
    preds = []
    for i in range(0, len(inputs), batch):
        batch_t = torch.cat(inputs[i:i+batch], dim=0)
        preds.extend(model(batch_t).argmax(dim=1).cpu().tolist())
    return preds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    true_labels = load_true_labels()
    print(f"True labels loaded: {len(true_labels)}")

    print("Loading images...")
    names, inputs = load_images(device)
    true_list = [true_labels[n] for n in names]

    from pytorchcv.model_provider import get_model

    results = []
    for mname, pool in ALL_MODELS:
        print(f"  [{pool}] {mname} ...", end=" ", flush=True)
        try:
            model = get_model(mname, pretrained=True).to(device).eval()
            preds = predict_all(model, inputs, device)
            del model
            correct = sum(p == t for p, t in zip(preds, true_list))
            acc = correct / len(true_list)
            print(f"acc={acc:.4f} ({correct}/500)")
            results.append((mname, pool, acc, correct, preds))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((mname, pool, None, None, None))

    print("\n" + "="*60)
    print(f"{'Model':<42} {'Pool':<10} {'Accuracy':>8}  {'Status'}")
    print("-"*60)
    perfect = []
    imperfect = []
    for mname, pool, acc, correct, _ in results:
        if acc is None:
            status = "ERROR"
            print(f"{mname:<42} {pool:<10} {'N/A':>8}  {status}")
        elif acc == 1.0:
            status = "PERFECT"
            perfect.append(mname)
            print(f"{mname:<42} {pool:<10} {acc:>8.4f}  {status}")
        else:
            errors = 500 - correct
            status = f"{errors} wrong"
            imperfect.append((mname, acc, errors))
            print(f"{mname:<42} {pool:<10} {acc:>8.4f}  {status}")

    print("="*60)
    print(f"\nPerfect models ({len(perfect)}): {perfect}")
    print(f"Imperfect models ({len(imperfect)}):")
    for mname, acc, errors in imperfect:
        print(f"  {mname}: {acc:.4f} ({errors} errors)")

if __name__ == "__main__":
    main()
