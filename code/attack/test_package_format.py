import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from package_methods import package_method


def test_submission_zip_has_explicit_images_directory(tmp_path):
    adv_dir = tmp_path / "adv"
    adv_dir.mkdir()
    from PIL import Image
    for i in range(500):
        Image.new("RGB", (32, 32), color=(i % 255, 0, 0)).save(adv_dir / f"{i}.png")

    out_zip = package_method("dummy", adv_dir, tmp_path / "out")

    names = zipfile.ZipFile(out_zip).namelist()
    assert "images/" in names
    assert "images/0.png" in names
    assert "images/499.png" in names
    assert sum(n.startswith("images/") and n.endswith(".png") for n in names) == 500
    assert "label.txt" in names


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as d:
        test_submission_zip_has_explicit_images_directory(Path(d))
    print("package format tests passed")
