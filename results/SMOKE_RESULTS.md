# CIFAR-10 Attack Smoke Results

Recorded: 2026-06-11

All smoke scores use local evaluation on the first 8 images unless noted.
Score is `100 * pool_ASR * mean_SSIM`.

## Summary Table

| Run | Attack pool / method | Eval pool | Limit | SSIM | ASR | Score | Takeaway |
|---|---|---:|---:|---:|---:|---:|---|
| `adv_diverse_mixed_smoke20_r4` | Diverse, `2 CNN + 2 ViT + 4 Robust`, steps=20, EOT=1 | robust | 8 | 0.9350 | 0.0000 | 0.0000 | Strong ViT/CNN transfer, no robust success. |
| `adv_diverse_mixed_smoke20_r4` | Diverse, `2 CNN + 2 ViT + 4 Robust`, steps=20, EOT=1 | vit | 8 | 0.9350 | 0.7083 | 66.2272 | Best ViT smoke so far. |
| `adv_diverse_mixed_smoke20_r4` | Diverse, `2 CNN + 2 ViT + 4 Robust`, steps=20, EOT=1 | holdout | 8 | 0.9350 | 0.4911 | 45.9138 | Useful for CNN/ViT stage, not robust stage. |
| `adv_pgd_mixed_smoke20_r4` | Plain PGD/IFGSM, `2 CNN + 2 ViT + 4 Robust`, steps=20, CE | robust | 8 | 0.9591 | 0.0000 | 0.0000 | Mixed average CE still dilutes robust gradients. |
| `adv_pgd_mixed_smoke20_r4` | Plain PGD/IFGSM, `2 CNN + 2 ViT + 4 Robust`, steps=20, CE | vit | 8 | 0.9591 | 0.3750 | 35.9656 | Worse than diverse for ViT. |
| `adv_pgd_mixed_smoke20_r4` | Plain PGD/IFGSM, `2 CNN + 2 ViT + 4 Robust`, steps=20, CE | holdout | 8 | 0.9591 | 0.6250 | 59.9427 | Strong CNN transfer despite robust failure. |
| `adv_pgd_robust4_smoke20` | Plain PGD/IFGSM, `4 Robust`, steps=20, CE | robust | 8 | 0.9517 | 0.1458 | 13.8787 | Best robust smoke so far; direction is promising but unstable. |
| `adv_pgd_robust4_smoke20` | Plain PGD/IFGSM, `4 Robust`, steps=20, CE | vit | 8 | 0.9517 | 0.1250 | 11.8960 | Robust-only does not transfer to ViT. |
| `adv_pgd_robust4_smoke20` | Plain PGD/IFGSM, `4 Robust`, steps=20, CE | holdout | 8 | 0.9517 | 0.1250 | 11.8960 | Robust-only does not transfer to CNN holdout. |
| `adv_robust_engstrom_diag8` | Plain PGD/IFGSM, `1 Robust` (`robust_engstrom`), steps=20, CE | robust | 8 | 0.9490 | 0.0417 | 3.9542 | Single robust source transfers poorly to robust pool. |
| `adv_robust_engstrom_diag8` | Plain PGD/IFGSM, `1 Robust` (`robust_engstrom`), steps=20, CE | vit | 8 | 0.9490 | 0.0417 | 3.9542 | Minimal ViT transfer. |
| `adv_robust_engstrom_diag8` | Plain PGD/IFGSM, `1 Robust` (`robust_engstrom`), steps=20, CE | holdout | 8 | 0.9490 | 0.0804 | 7.6260 | Minimal CNN transfer. |
| `adv_robust_engstrom_diag8` | Plain PGD/IFGSM, `1 Robust` (`robust_engstrom`), steps=20, CE | surrogate | 8 | 0.9490 | 0.1042 | 9.8855 | Sanity only. |
| `adv_ilpd_mixed_smoke20_r4` | ILPD, `2 CNN + 2 ViT + 4 Robust`, steps=20 | robust | 8 | 0.9382 | 0.0000 | 0.0000 | No robust success. |
| `adv_ilpd_mixed_smoke20_r4` | ILPD, `2 CNN + 2 ViT + 4 Robust`, steps=20 | vit | 8 | 0.9382 | 0.3750 | 35.1813 | Similar to mixed PGD on ViT, below diverse. |
| `adv_ilpd_mixed_smoke20_r4` | ILPD, `2 CNN + 2 ViT + 4 Robust`, steps=20 | holdout | 8 | 0.9382 | 0.7500 | 70.3626 | Best holdout-CNN smoke, but robust remains zero. |

## Notes

- The best robust smoke is `adv_pgd_robust4_smoke20`, but its robust ASR is still only `0.1458`.
- The best ViT smoke is `adv_diverse_mixed_smoke20_r4`, with ViT score `66.2272`.
- The best holdout-CNN smoke is `adv_ilpd_mixed_smoke20_r4`, with holdout score `70.3626`.
- Mixed `2 CNN + 2 ViT + 4 Robust` average-gradient methods have repeatedly failed on robust models.
- Robust-only PGD improves robust score but sacrifices ViT/CNN transfer, suggesting a staged attack may be necessary.

## Partial Full Run

`adv_pgd_robust4_full20` is not a clean completed run and should not be used for comparison:

- First attempt OOMed after saving 51 images.
- Resume attempt advanced the directory to 81 images.
- The resumed process stopped writing new PNGs after `80.png` at 2026-06-11 20:51:16 while still consuming GPU/CPU.
- Because it is partial and involved a mid-run diagnostic-logging change, do not treat it as a final result.
