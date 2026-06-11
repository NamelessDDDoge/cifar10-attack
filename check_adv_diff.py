import numpy as np
from PIL import Image
from pathlib import Path

clean_dir = Path(r"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\workspace\data\images")
adv_dir   = Path(r"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\workspace\results\adv_ilpd")

print(f"adv_dir exists: {adv_dir.exists()}")
print(f"adv png count: {len(list(adv_dir.glob('*.png')))}")

max_diffs = []
for i in range(20):
    c = np.array(Image.open(clean_dir / f"{i}.png").convert("RGB"), dtype=np.float32)
    a = np.array(Image.open(adv_dir   / f"{i}.png").convert("RGB"), dtype=np.float32)
    d = np.abs(a - c)
    max_diffs.append(d.max())
    print(f"  {i}.png  max_diff={d.max():.2f}  mean_diff={d.mean():.4f}  (eps=8/255 → 8 on [0,255] scale)")

print(f"\nOverall max (first 20): {max(max_diffs):.2f}")
print(f"Any diff > 0: {any(m > 0 for m in max_diffs)}")
