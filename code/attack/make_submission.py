# -*- coding: utf-8 -*-
"""
Pack adversarial images into submission.zip in competition format:
  images/0.png ... images/499.png
  label.txt

Default is LOSSLESS PNG passthrough. JPEG re-encoding is destructive to
adversarial perturbations (lowers ASR and changes SSIM) — only use --jpeg
if the competition explicitly requires .jpg.

Prefer the project-root packager (package_submission.py) which also
validates image size/mode and writes a submission report.
"""
import argparse, io, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ADV_IMAGES_DIR, LABEL_FILE, RESULTS_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--adv-dir",   default=str(ADV_IMAGES_DIR))
parser.add_argument("--label-txt", default=str(LABEL_FILE))
parser.add_argument("--out-zip",   default=str(RESULTS_DIR / "submission.zip"))
parser.add_argument("--jpeg", action="store_true",
                    help="re-encode as JPEG (DESTRUCTIVE to perturbations; off by default)")
parser.add_argument("--quality",   type=int, default=95)
args = parser.parse_args()

ADV_DIR   = Path(args.adv_dir)
LABEL_TXT = Path(args.label_txt)
OUT_ZIP   = Path(args.out_zip)

pngs = sorted(ADV_DIR.glob("*.png"), key=lambda p: int(p.stem))
assert len(pngs) == 500, f"Expected 500 pngs, got {len(pngs)}"

OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("images/", "")
    for png in pngs:
        if args.jpeg:
            from PIL import Image
            img = Image.open(png).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=args.quality)
            zf.writestr(f"images/{png.stem}.jpg", buf.getvalue())
        else:
            zf.write(png, f"images/{png.name}")
    zf.write(LABEL_TXT, "label.txt")

entries = zipfile.ZipFile(OUT_ZIP).namelist()
print(f"Done: {OUT_ZIP}")
print(f"  format: {'JPEG q=' + str(args.quality) if args.jpeg else 'lossless PNG'}")
print(f"  total entries: {len(entries)}")
print(f"  sample: {entries[:3]}")
