"""ATT runner for the local CIFAR-10 transfer-attack benchmark.

This adapts the NeurIPS 2024 Adaptive Token Tuning implementation already
vendored under workspace/code/repos/TransferAttack. The source ATT model is a
224x224 ImageNet ViT, while this benchmark submits 32x32 CIFAR-10 PNGs. We
therefore optimize at 224x224, downsample the perturbation, then project again
around the original 32x32 image before saving.

Important isolation rule: this script does not import eval/external_models.py.
External/holdout models are only used later by workspace/code/eval/evaluate.py.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EPS, RESULTS_DIR, REPOS_DIR
from run_utils import (
    setup_logger,
    load_data,
    adv_to_numpy,
    save_adv_batch,
    get_done_set,
    image_pbar,
    step_pbar,
    iter_todo_batches,
    log_progress_event,
)

REPO_DIR = REPOS_DIR / "TransferAttack"
sys.path.insert(0, str(REPO_DIR))

TAG = "ATT"
DEFAULT_ADV_DIR = RESULTS_DIR / "adv_att"
DEFAULT_MODEL = "vit_base_patch16_224"
DEFAULT_STEPS = 10
DEFAULT_BATCH_SIZE = 1


def _patch_att_forward(att_module):
    """Bind the vendored module-level forward function as ATT.forward.

    The local vendored file currently defines `forward` at module scope. This
    runner keeps the third-party file untouched and attaches a corrected method
    at runtime.
    """

    def forward(self, data, label, sbar=None, **kwargs):
        if self.targeted:
            raise ValueError("run_att.py supports untargeted ATT only")

        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        momentum = 0
        delta = self.init_delta(data)
        delta.requires_grad_()

        # Warm-up backward pass for feature-gradient patch importance.
        self.model.zero_grad(set_to_none=True)
        output = self.model(data + delta)
        output.backward(torch.ones_like(output))

        resize = att_module.transforms.Resize((224, 224))
        gf = (self.im_fea[0][1:] * self.im_grad[0][1:]).sum(-1)
        gf = resize(gf.reshape(1, 14, 14))
        gf_patchs_t = self.norm_patchs(gf, self.patch_index, self.size, self.scale, self.offset)
        gf_patchs_start = torch.ones_like(gf_patchs_t, device=self.device) * 0.99
        gf_offset = (gf_patchs_start - gf_patchs_t) / self.epoch

        for i in range(self.epoch):
            self.var_A = 0
            self.var_qkv = 0
            self.var_mlp = 0
            self.back_attn = 11
            torch.manual_seed(i)

            random_patch = (
                torch.rand(14, 14, device=self.device)
                .repeat_interleave(16)
                .reshape(14, 14 * 16)
                .repeat(1, 16)
                .reshape(224, 224)
            )
            threshold = gf_patchs_start - gf_offset * (i + 1)
            gf_patchs = torch.where(random_patch > threshold, 0.0, 1.0).to(self.device)

            outputs = self.get_logits(data + delta * gf_patchs.detach())
            loss = self.get_loss(outputs, label)
            grad = self.get_grad(loss, delta)
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, data, momentum, self.alpha)

            if sbar is not None:
                sbar.update(1)
                sbar.set_postfix(loss=f"{float(loss.detach().cpu()):.3f}")

        return delta.detach()

    att_module.ATT.forward = forward


def build_attacker(args, device, logger):
    if device.type != "cuda":
        raise RuntimeError("The vendored ATT implementation currently requires CUDA")

    from transferattack.model_related import att as att_module

    _patch_att_forward(att_module)
    logger.info(f"Loading ATT source model {args.model_name}")
    attacker = att_module.ATT(
        model_name=args.model_name,
        epsilon=args.eps,
        alpha=args.alpha,
        epoch=args.steps,
        decay=args.decay,
        targeted=False,
        random_start=args.random_start,
        norm="linfty",
        loss="crossentropy",
        device=device,
        lam=args.lam,
        sample_num_batches=args.sample_num_batches,
    )
    return attacker


@torch.no_grad()
def choose_attack_labels(attacker, images_224, true_labels, mode):
    logits = attacker.model(images_224)
    if mode == "true":
        if logits.shape[1] <= int(true_labels.max().item()):
            raise ValueError(
                f"true label mode incompatible with source logits dim={logits.shape[1]}"
            )
        return true_labels.clone(), logits.argmax(dim=1)
    if mode == "self":
        pred = logits.argmax(dim=1)
        return pred.clone(), pred
    if mode == "auto":
        if logits.shape[1] == 10:
            return true_labels.clone(), logits.argmax(dim=1)
        pred = logits.argmax(dim=1)
        return pred.clone(), pred
    raise ValueError(f"unknown label mode: {mode}")


def project_downsampled_delta(clean_32, delta_224, eps):
    delta_32 = F.interpolate(delta_224, size=(32, 32), mode="bilinear", align_corners=False)
    delta_32 = delta_32.clamp(-eps, eps)
    return (clean_32 + delta_32).clamp(0, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ATT on local CIFAR-10 PNG benchmark")
    parser.add_argument("--out-dir", default=str(DEFAULT_ADV_DIR))
    parser.add_argument("--limit", "--max-images", dest="limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--sample-num-batches", type=int, default=130)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--label-mode", choices=["auto", "true", "self"], default="auto")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int, default=20260611)
    return parser.parse_args()


def main():
    args = parse_args()
    args.alpha = args.alpha if args.alpha is not None else args.eps / args.steps

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    logger = setup_logger(TAG)
    adv_dir = Path(args.out_dir)
    done_set = get_done_set(adv_dir)

    limit = args.limit if args.limit is not None else 500
    if not 1 <= limit <= 500:
        raise ValueError("--limit must be in [1, 500]")

    if args.limit is None and len(done_set) == 500:
        logger.info("Already complete (500 files). Skip.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device}")
    logger.info(f"out_dir={adv_dir}")
    logger.info(
        f"model={args.model_name} steps={args.steps} eps={args.eps:.8f} "
        f"alpha={args.alpha:.8f} label_mode={args.label_mode}"
    )

    attacker = build_attacker(args, device, logger)

    logger.info("Loading data and true labels")
    names, images, true_labels = load_data(device)
    todo = [(i, n) for i, n in enumerate(names[:limit]) if n not in done_set]
    adv_dir.mkdir(parents=True, exist_ok=True)
    history_path = adv_dir / "attack_history.jsonl"
    logger.info(f"Images to process: {len(todo)}/{limit}")

    with image_pbar(limit, min(len(done_set), limit), TAG) as ibar:
        for chunk in iter_todo_batches(todo, args.batch_size):
            idxs, names_b = zip(*chunk)
            idxs = list(idxs)
            clean_32 = images[idxs]
            clean_224 = F.interpolate(clean_32, size=(224, 224), mode="bilinear", align_corners=False)
            true_b = true_labels[idxs]
            attack_labels, source_pred = choose_attack_labels(
                attacker, clean_224, true_b, args.label_mode
            )

            batch_desc = f"[{idxs[0]}:{idxs[-1] + 1}]"
            with step_pbar(args.steps, batch_desc) as sbar:
                delta_224 = attacker(clean_224, attack_labels, sbar=sbar)

            adv_32 = project_downsampled_delta(clean_32, delta_224, args.eps)
            save_adv_batch(adv_to_numpy(adv_32), list(names_b), adv_dir)

            max_linf = float((adv_32 - clean_32).abs().amax().detach().cpu())
            rec = {
                "batch": [int(idxs[0]), int(idxs[-1])],
                "names": list(names_b),
                "model_name": args.model_name,
                "label_mode": args.label_mode,
                "true_labels": [int(v) for v in true_b.detach().cpu()],
                "attack_labels": [int(v) for v in attack_labels.detach().cpu()],
                "source_predictions": [int(v) for v in source_pred.detach().cpu()],
                "eps": float(args.eps),
                "alpha": float(args.alpha),
                "steps": int(args.steps),
                "max_linf": max_linf,
            }
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            ibar.update(len(chunk))
            log_progress_event(
                logger, TAG, adv_dir, len(get_done_set(adv_dir)),
                total=limit, batch=[int(idxs[0]), int(idxs[-1])], max_linf=max_linf,
            )
            logger.info(f"Saved batch {batch_desc}; max_linf={max_linf:.6f}")

    logger.info(f"Done. {len(list(adv_dir.glob('*.png')))} PNGs in {adv_dir}")


if __name__ == "__main__":
    main()
