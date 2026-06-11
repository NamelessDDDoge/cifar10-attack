# -*- coding: utf-8 -*-
"""Validate and package each attack output directory as an independent submission."""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LABEL_FILE, RESULTS_DIR, ATTACK_DIR


METHOD_DIRS = {
    "ilpd": RESULTS_DIR / "adv_ilpd",
    "pgn": RESULTS_DIR / "adv_pgn",
    "bsr": RESULTS_DIR / "adv_bsr",
    "sit": RESULTS_DIR / "adv_sit",
    "awt": RESULTS_DIR / "adv_awt",
    "l2t": RESULTS_DIR / "adv_l2t",
    "diverse_vit": RESULTS_DIR / "adv_diverse_vit",
}


RUN_COMMANDS = {
    "ilpd": [sys.executable, "-u", str(ATTACK_DIR / "run_ilpd.py")],
    "pgn": [sys.executable, "-u", str(ATTACK_DIR / "run_pgn.py")],
    "bsr": [sys.executable, "-u", str(ATTACK_DIR / "run_bsr.py")],
    "sit": [sys.executable, "-u", str(ATTACK_DIR / "run_sit.py")],
    "awt": [sys.executable, "-u", str(ATTACK_DIR / "run_awt.py")],
    "l2t": [sys.executable, "-u", str(ATTACK_DIR / "run_l2t.py")],
    "diverse_vit": [
        sys.executable, "-u", str(ATTACK_DIR / "run_diverse.py"),
        "--out-dir", str(RESULTS_DIR / "adv_diverse_vit"),
        "--steps", "50",
        "--batch-size", "2",
        "--cnn-count", "4",
        "--vit-surrogates", "all",
        "--eot", "2",
    ],
}


def safe_print(text: str = "") -> None:
    data = (text + ("" if text.endswith("\n") else "\n")).encode(
        sys.stdout.encoding or "utf-8", errors="backslashreplace")
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def pngs_for(adv_dir: Path) -> list[Path]:
    return sorted(adv_dir.glob("*.png"), key=lambda p: int(p.stem))


def validate_adv_dir(adv_dir: Path) -> None:
    if not adv_dir.is_dir():
        raise RuntimeError(f"{adv_dir} is not a directory")
    pngs = pngs_for(adv_dir)
    if len(pngs) != 500:
        raise RuntimeError(f"{adv_dir}: expected 500 PNGs, got {len(pngs)}")
    names = {p.name for p in pngs}
    for i in range(500):
        name = f"{i}.png"
        if name not in names:
            raise RuntimeError(f"{adv_dir}: missing {name}")
        with Image.open(adv_dir / name) as img:
            if img.size != (32, 32):
                raise RuntimeError(f"{adv_dir}/{name}: expected 32x32, got {img.size}")
            if img.mode != "RGB":
                raise RuntimeError(f"{adv_dir}/{name}: expected RGB, got {img.mode}")


def package_method(method: str, adv_dir: Path, out_dir: Path) -> Path:
    validate_adv_dir(adv_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / f"submission_{method}.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("images/", "")
        for i in range(500):
            name = f"{i}.png"
            zf.write(adv_dir / name, arcname=f"images/{name}")
        zf.write(LABEL_FILE, arcname="label.txt")
    return out_zip


def run_method(method: str, log_dir: Path) -> int:
    cmd = RUN_COMMANDS[method]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{method}.log"
    safe_print(f"[run] {method}: {' '.join(cmd)}")
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n=== RUN {method} ===\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            safe_print(line)
            log.write(line)
            log.flush()
        return proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHOD_DIRS),
                        choices=list(METHOD_DIRS))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "submissions_by_method"))
    parser.add_argument("--log-dir", default=str(RESULTS_DIR / "full_run_logs"))
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    manifest = {}
    for method in args.methods:
        adv_dir = METHOD_DIRS[method]
        if not args.skip_run and len(pngs_for(adv_dir)) != 500:
            code = run_method(method, log_dir)
            if code != 0:
                raise SystemExit(f"{method} failed with exit code {code}")
        else:
            safe_print(f"[skip-run] {method}: {len(pngs_for(adv_dir))}/500 PNGs present")

        out_zip = package_method(method, adv_dir, out_dir)
        manifest[method] = {
            "adv_dir": str(adv_dir),
            "zip": str(out_zip),
            "png_count": len(pngs_for(adv_dir)),
        }
        safe_print(f"[package] {method}: {out_zip}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    safe_print(f"[done] wrote {manifest_path}")


if __name__ == "__main__":
    main()
