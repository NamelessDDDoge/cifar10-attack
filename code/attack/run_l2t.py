"""
L2T (Learning to Transform) attack adapter for CIFAR-10
Origin: https://github.com/RongyiZhu/L2T
Paper: Zhu et al., CVPR 2024

Adaptations from original l2t.py / main.py:
  - Overrides load_model to use pytorchcv CIFAR-10 surrogate ensemble
  - eps=8/255, alpha=2/255, epoch=10, num_scale=3
  - Truncate op_list to exclude ssm/crop ops that hardcode 224x224 image size
  - PNG load/save for 32x32 images
  - Removed ImageNet loader; no CUDA requirement
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, REPOS_DIR, EPS, ALPHA, BATCH_SIZE
from run_utils import (
    setup_logger, load_data, build_l2t_ensemble, adv_to_numpy,
    save_adv_batch, get_done_set, image_pbar,
    iter_todo_batches, log_progress_event,
)

REPO_DIR = REPOS_DIR / "L2T"
sys.path.insert(0, str(REPO_DIR))

TAG     = "L2T"
ADV_DIR = RESULTS_DIR / "adv_l2t"

EPOCH           = 10
NUM_SCALE       = 2   # L2T multiplies batch×num_scale per forward; keep ≤3 on large models
L2T_BATCH_SIZE  = max(1, BATCH_SIZE // NUM_SCALE)  # effective batch stays ≈ BATCH_SIZE


def main():
    logger   = setup_logger(TAG)
    done_set = get_done_set(ADV_DIR)

    if len(done_set) == 500:
        logger.info("Already complete (500 files). Skip.")
        return

    logger.info(f"Resume: {len(done_set)}/500 done")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    logger.info("Building L2T ensemble (light models only to fit 8GB VRAM) ...")
    ensemble = build_l2t_ensemble(device, TAG, logger)

    logger.info("Loading data ...")
    names, images, labels = load_data(device)

    import l2t as l2t_module
    logger.info(f"Truncating op_list: {len(l2t_module.op_list)} -> 71 ops (removing ssm/crop/affine)")
    l2t_module.op_list = l2t_module.op_list[:71]

    class CifarL2T(l2t_module.L2T):
        def load_model(self, model_name):
            return ensemble

    attacker = CifarL2T(
        model_name="cifar10_ensemble",
        epsilon=EPS, alpha=ALPHA,
        epoch=EPOCH, decay=1.0,
        targeted=False, random_start=False,
        norm="linfty", loss="crossentropy",
        device=device, num_scale=NUM_SCALE,
    )

    ADV_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(i, n) for i, n in enumerate(names) if n not in done_set]
    logger.info(f"Images to process: {len(todo)}")

    logger.info(f"L2T_BATCH_SIZE={L2T_BATCH_SIZE} (effective={L2T_BATCH_SIZE*NUM_SCALE} after transform)")
    with image_pbar(500, len(done_set), TAG) as ibar:
        for chunk in iter_todo_batches(todo, L2T_BATCH_SIZE):
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
