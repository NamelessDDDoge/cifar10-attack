"""
SIA (Structure Invariant Attack / SIT) adapter for CIFAR-10
Origin: https://github.com/xiaosen-wang/SIT
Paper: Wang et al., ICCV 2023

Adaptations from original attack.py / main.py:
  - Pass pytorchcv CIFAR-10 surrogate ensemble directly as model (SIA accepts any model)
  - eps=8/255, alpha=2/255, epoch=10
  - num_block=4 (32x32/8=4 blocks per axis)
  - num_copies=10 (reduced from 20 for CPU speed)
  - PNG load/save for 32x32 images
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, REPOS_DIR, EPS, ALPHA, BATCH_SIZE
from run_utils import (
    setup_logger, load_data, build_ensemble, adv_to_numpy,
    save_adv_batch, get_done_set, image_pbar,
    iter_todo_batches, log_progress_event,
)

REPO_DIR = REPOS_DIR / "SIT"
sys.path.insert(0, str(REPO_DIR))

TAG     = "SIT"
ADV_DIR = RESULTS_DIR / "adv_sit"

EPOCH      = 10
NUM_COPIES = 20
NUM_BLOCK  = 4


def main():
    logger   = setup_logger(TAG)
    done_set = get_done_set(ADV_DIR)

    if len(done_set) == 500:
        logger.info("Already complete (500 files). Skip.")
        return

    logger.info(f"Resume: {len(done_set)}/500 done")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    logger.info("Building surrogate ensemble ...")
    ensemble = build_ensemble(device, TAG, logger)

    logger.info("Loading data ...")
    names, images, labels = load_data(device)

    from attack import SIA

    attacker = SIA(
        model=ensemble,
        epsilon=EPS, alpha=ALPHA,
        epoch=EPOCH, decay=1.0,
        num_copies=NUM_COPIES, num_block=NUM_BLOCK,
        targeted=False, random_start=False,
        norm="linfty", loss="crossentropy", device=device,
    )

    ADV_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(i, n) for i, n in enumerate(names) if n not in done_set]
    logger.info(f"Images to process: {len(todo)}")

    with image_pbar(500, len(done_set), TAG) as ibar:
        for chunk in iter_todo_batches(todo, BATCH_SIZE):
            idxs, names_b = zip(*chunk)
            idxs = list(idxs)

            img_b  = images[idxs]
            lbl_b  = labels[idxs]
            delta  = attacker(img_b, lbl_b)
            adv_b  = torch.clamp(img_b + delta, 0, 1).detach()

            save_adv_batch(adv_to_numpy(adv_b), list(names_b), ADV_DIR)
            ibar.update(len(chunk))
            log_progress_event(logger, TAG, ADV_DIR, len(get_done_set(ADV_DIR)),
                               batch=[int(idxs[0]), int(idxs[-1])])
            logger.debug(f"Saved batch [{idxs[0]}:{idxs[-1]+1}] ({len(chunk)} imgs)")

    logger.info(f"Done. {len(list(ADV_DIR.glob('*.png')))} images in {ADV_DIR}")


if __name__ == "__main__":
    main()
