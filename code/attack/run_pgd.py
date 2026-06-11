"""Plain IFGSM/PGD-style CIFAR-10 attack over mixed surrogate pools.

This runner intentionally avoids transfer tricks such as DI/SI/TI, momentum,
and EOT. It is a diagnostic baseline for robust models: if this cannot move
white-box robust surrogates, more elaborate transfer transforms are unlikely
to help.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, EPS, ALPHA, BATCH_SIZE
from run_utils import (
    setup_logger, load_data, adv_to_numpy, save_adv_batch,
    get_done_set, image_pbar, step_pbar,
    iter_todo_batches, log_progress_event,
)
from run_diverse import build_attack_models, untargeted_margin_loss


TAG = "PGD"
DEFAULT_ADV_DIR = RESULTS_DIR / "adv_pgd"


def attack_pool_asr(models, x, labels):
    vals = []
    with torch.no_grad():
        for am in models:
            vals.append((am.model(x).argmax(dim=1) != labels).float().mean())
    return torch.stack(vals)


def pgd_loss(models, x, labels, loss_name):
    loss = x.new_tensor(0.0)
    for am in models:
        logits = am.model(x)
        if loss_name == "ce":
            loss = loss + F.cross_entropy(logits, labels)
        elif loss_name == "margin":
            loss = loss + untargeted_margin_loss(logits, labels)
        else:
            raise ValueError(f"unknown loss: {loss_name}")
    return loss / len(models)


def pgd_attack(
    models,
    images,
    labels,
    steps,
    step_size,
    epsilon,
    loss_name,
    random_start=False,
    asr_every=1,
    sbar=None,
):
    x0 = images.detach()
    if random_start:
        x = (x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon)).clamp(0, 1)
    else:
        x = x0.clone()

    history = []
    for step in range(steps):
        x_req = x.detach().requires_grad_(True)
        loss = pgd_loss(models, x_req, labels, loss_name)
        grad = torch.autograd.grad(loss, x_req)[0].detach()
        x = x.detach() + step_size * torch.sign(grad)
        x = torch.min(torch.max(x, x0 - epsilon), x0 + epsilon).clamp(0, 1)

        should_eval_asr = asr_every > 0 and ((step + 1) % asr_every == 0 or step + 1 == steps)
        asr = attack_pool_asr(models, x, labels) if should_eval_asr else None
        rec = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "pool_asr": float(asr.mean().detach().cpu()) if asr is not None else None,
            "asr": [float(v) for v in asr.detach().cpu()] if asr is not None else None,
        }
        history.append(rec)
        if sbar is not None:
            sbar.update(1)
            postfix = {"loss": f"{rec['loss']:.3f}"}
            if rec["pool_asr"] is not None:
                postfix["pool_asr"] = f"{rec['pool_asr']:.2f}"
            sbar.set_postfix(**postfix)

    return x.detach(), history


def parse_args():
    parser = argparse.ArgumentParser(description="Plain mixed-pool IFGSM/PGD attack")
    parser.add_argument("--out-dir", default=str(DEFAULT_ADV_DIR))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=min(2, BATCH_SIZE))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--cnn-count", type=int, default=2)
    parser.add_argument("--vit-surrogates", default="none")
    parser.add_argument("--robust-surrogates", default="none")
    parser.add_argument("--loss", choices=["ce", "margin"], default="ce")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument(
        "--asr-every",
        type=int,
        default=1,
        help="Evaluate attack-pool ASR every N steps; 0 disables in-attack ASR logging",
    )
    parser.add_argument("--seed", type=int, default=20260611)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    adv_dir = Path(args.out_dir)
    logger = setup_logger(TAG)
    done_set = get_done_set(adv_dir)
    limit = args.max_images if args.max_images is not None else 500
    if args.max_images is None and len(done_set) == 500:
        logger.info("Already complete (500 files). Skip.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")
    logger.info(f"out_dir={adv_dir}")
    logger.info(
        f"steps={args.steps} batch_size={args.batch_size} loss={args.loss} "
        f"random_start={args.random_start} asr_every={args.asr_every}"
    )
    logger.info(
        f"cnn_count={args.cnn_count} vit_surrogates={args.vit_surrogates} "
        f"robust_surrogates={args.robust_surrogates}"
    )

    models = build_attack_models(
        device,
        logger,
        cnn_count=args.cnn_count,
        vit_selection=args.vit_surrogates,
        robust_selection=args.robust_surrogates,
    )
    names, images, labels = load_data(device)

    todo = [(i, n) for i, n in enumerate(names[:limit]) if n not in done_set]
    logger.info(f"Images to process: {len(todo)}/{limit}")
    adv_dir.mkdir(parents=True, exist_ok=True)

    history_path = adv_dir / "attack_history.jsonl"
    with image_pbar(limit, min(len(done_set), limit), TAG) as ibar:
        for chunk in iter_todo_batches(todo, args.batch_size):
            idxs, names_b = zip(*chunk)
            idxs = list(idxs)
            batch_desc = f"[{idxs[0]}:{idxs[-1] + 1}]"
            with step_pbar(args.steps, batch_desc) as sbar:
                adv_b, hist = pgd_attack(
                    models=models,
                    images=images[idxs],
                    labels=labels[idxs],
                    steps=args.steps,
                    step_size=ALPHA,
                    epsilon=EPS,
                    loss_name=args.loss,
                    random_start=args.random_start,
                    asr_every=args.asr_every,
                    sbar=sbar,
                )

            save_adv_batch(adv_to_numpy(adv_b), list(names_b), adv_dir)
            with history_path.open("a", encoding="utf-8") as f:
                for rec in hist:
                    rec["batch"] = [int(idxs[0]), int(idxs[-1])]
                    rec["models"] = [m.name for m in models]
                    f.write(json.dumps(rec) + "\n")
            ibar.update(len(chunk))
            log_progress_event(
                logger, TAG, adv_dir, len(get_done_set(adv_dir)),
                total=limit, batch=[int(idxs[0]), int(idxs[-1])],
                train_pool_asr=hist[-1]["pool_asr"] if hist else None,
            )
            train_pool_asr = hist[-1]["pool_asr"] if hist else None
            if train_pool_asr is None:
                logger.info(f"Saved batch {batch_desc}; train_pool_asr=not_evaluated")
            else:
                logger.info(f"Saved batch {batch_desc}; train_pool_asr={train_pool_asr:.4f}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    logger.info(f"Done. {len(list(adv_dir.glob('*.png')))} PNGs in {adv_dir}")


if __name__ == "__main__":
    main()
