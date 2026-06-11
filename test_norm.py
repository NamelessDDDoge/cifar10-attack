import torch, numpy as np, sys
from pytorchcv.model_provider import get_model
from PIL import Image
from pathlib import Path

IMAGES = Path(r"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\workspace\data\images")
LABELS = Path(r"C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\workspace\data\label.txt")

labels = {}
with open(LABELS) as f:
    for line in f:
        n,c = line.strip().split()
        labels[n]=int(c)

names = [f"{i}.png" for i in range(100)]
imgs = []
for n in names:
    arr = np.array(Image.open(IMAGES/n).convert("RGB"),dtype=np.float32)/255.0
    imgs.append(torch.from_numpy(arr.transpose(2,0,1)))
X = torch.stack(imgs)
y = torch.tensor([labels[n] for n in names])

mean=torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
std=torch.tensor([0.2023,0.1994,0.2010]).view(1,3,1,1)
Xn=(X-mean)/std

for mname in ["wrn16_10_cifar10", "pyramidnet164_a270_bn_cifar10", "seresnet110_cifar10"]:
    m = get_model(mname, pretrained=True).eval()
    with torch.no_grad():
        acc_raw  = (m(X).argmax(1)==y).float().mean().item()
        acc_norm = (m(Xn).argmax(1)==y).float().mean().item()
    print(f"{mname}: acc_raw={acc_raw:.3f}  acc_norm={acc_norm:.3f}", flush=True)
