"""
BSR (Block Shuffle and Rotation) attack adapter for CIFAR-10
Origin: TransferAttack (https://github.com/Trustworthy-AI-Group/TransferAttack)
Paper: Wang et al., CVPR 2024 (arxiv:2308.10299)

Adaptations:
  - Override load_model to use pytorchcv CIFAR-10 surrogate ensemble
  - eps=8/255, alpha=2/255, epoch=10
  - num_scale=10, num_block=3 (32x32/block ~= 10 pixels per block)
  - PNG load/save for 32x32 images
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, REPOS_DIR, EPS, ALPHA, BATCH_SIZE
from run_utils import (
    setup_logger, load_data, build_ensemble, build_mixed_attack_ensemble, adv_to_numpy,
    save_adv_batch, get_done_set, image_pbar,
    iter_todo_batches, log_progress_event,
)

REPO_DIR = REPOS_DIR / "TransferAttack"
sys.path.insert(0, str(REPO_DIR))

TAG     = "BSR"
ADV_DIR = RESULTS_DIR / "adv_bsr"

EPOCH     = 10
NUM_SCALE = 20
NUM_BLOCK = 3
DECAY     = 1.0


def main():
    parser = argparse.ArgumentParser(description="BSR CIFAR-10 attack")
    parser.add_argument("--out-dir", default=str(ADV_DIR))
    parser.add_argument("--steps", type=int, default=EPOCH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--cnn-count", type=int, default=None)
    parser.add_argument("--vit-surrogates", default="none")
    parser.add_argument("--robust-surrogates", default="none")
    args = parser.parse_args()

    adv_dir = Path(args.out_dir)
    logger   = setup_logger(TAG)
    done_set = get_done_set(adv_dir)

    limit = args.max_images if args.max_images is not None else 500
    if args.max_images is None and len(done_set) == 500:
        logger.info("Already complete (500 files). Skip.")
        return

    logger.info(f"Resume: {len(done_set)}/{limit} done")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")

    logger.info("Building surrogate ensemble ...")
    if args.cnn_count is None and args.vit_surrogates == "none" and args.robust_surrogates == "none":
        ensemble = build_ensemble(device, TAG, logger)
    else:
        ensemble = build_mixed_attack_ensemble(
            device, TAG, logger,
            cnn_count=args.cnn_count or 0,
            vit_surrogates=args.vit_surrogates,
            robust_surrogates=args.robust_surrogates,
        )

    logger.info("Loading data ...")
    names, images, labels = load_data(device)

    from transferattack.input_transformation.bsr import BSR

    class CifarBSR(BSR):
        def load_model(self, model_name):
            return ensemble

    attacker = CifarBSR(
        model_name="cifar10_ensemble",
        epsilon=EPS, alpha=ALPHA,
        epoch=args.steps, decay=DECAY,
        num_scale=NUM_SCALE, num_block=NUM_BLOCK,
        targeted=False, random_start=False,
        norm="linfty", loss="crossentropy", device=device,
    )

    adv_dir.mkdir(parents=True, exist_ok=True)
    todo = [(i, n) for i, n in enumerate(names[:limit]) if n not in done_set]
    logger.info(f"Images to process: {len(todo)}")

    with image_pbar(limit, min(len(done_set), limit), TAG) as ibar:
        for chunk in iter_todo_batches(todo, args.batch_size):
            idxs, names_b = zip(*chunk)
            idxs = list(idxs)

            img_b  = images[idxs]
            lbl_b  = labels[idxs]
            delta  = attacker(img_b, lbl_b)
            adv_b  = torch.clamp(img_b + delta, 0, 1).detach()

            save_adv_batch(adv_to_numpy(adv_b), list(names_b), adv_dir)
            ibar.update(len(chunk))
            log_progress_event(logger, TAG, adv_dir, len(get_done_set(adv_dir)),
                               total=limit, batch=[int(idxs[0]), int(idxs[-1])])
            logger.debug(f"Saved batch [{idxs[0]}:{idxs[-1]+1}] ({len(chunk)} imgs)")

    logger.info(f"Done. {len(list(adv_dir.glob('*.png')))} images in {adv_dir}")


if __name__ == "__main__":
    main()
