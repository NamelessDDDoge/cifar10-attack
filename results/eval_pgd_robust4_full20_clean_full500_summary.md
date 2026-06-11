# PGD Robust4 Full20 Clean - Full 500 Evaluation

Adversarial images: `workspace/results/adv_pgd_robust4_full20_clean`

Scoring formula: `Score = 100 * ASR * mean_SSIM`, where `mean_SSIM = 0.9585` for this 500-image set.

## Category Summary

| Category | Models | n | ASR | Attacked / Total | Score |
|---|---:|---:|---:|---:|---:|
| CNN | 20 | 500 | 0.1728 | 1728/10000 | 16.5620 |
| ViT | 3 | 500 | 0.1413 | 212/1500 | 13.5461 |
| Robust | 6 | 500 | 0.1930 | 579/3000 | 18.4981 |

## Per-Model Results

| Model | Category | ASR | Attacked / Total | Score |
|---|---|---:|---:|---:|
| `pyramidnet164_a270_bn_cifar10` | CNN | 0.1420 | 71/500 | 13.6100 |
| `wrn16_10_cifar10` | CNN | 0.1640 | 82/500 | 15.7186 |
| `densenet190_k40_bc_cifar10` | CNN | 0.1520 | 76/500 | 14.5684 |
| `seresnet110_cifar10` | CNN | 0.2000 | 100/500 | 19.1690 |
| `resnext29_16x64d_cifar10` | CNN | 0.1380 | 69/500 | 13.2266 |
| `diaresnet56_cifar10` | CNN | 0.2320 | 116/500 | 22.2360 |
| `pyramidnet236_a220_bn_cifar10` | CNN | 0.1480 | 74/500 | 14.1851 |
| `pyramidnet110_a270_cifar10` | CNN | 0.1640 | 82/500 | 15.7186 |
| `pyramidnet272_a200_bn_cifar10` | CNN | 0.1660 | 83/500 | 15.9103 |
| `densenet250_k24_bc_cifar10` | CNN | 0.1600 | 80/500 | 15.3352 |
| `pyramidnet110_a84_cifar10` | CNN | 0.1980 | 99/500 | 18.9773 |
| `wrn20_10_1bit_cifar10` | CNN | 0.1840 | 92/500 | 17.6355 |
| `wrn28_10_cifar10` | CNN | 0.1500 | 75/500 | 14.3768 |
| `resnet272bn_cifar10` | CNN | 0.2160 | 108/500 | 20.7025 |
| `pyramidnet200_a240_bn_cifar10` | CNN | 0.1520 | 76/500 | 14.5684 |
| `resnext272_1x64d_cifar10` | CNN | 0.1660 | 83/500 | 15.9103 |
| `seresnet164bn_cifar10` | CNN | 0.2000 | 100/500 | 19.1690 |
| `wrn20_10_32bit_cifar10` | CNN | 0.1740 | 87/500 | 16.6770 |
| `wrn40_8_cifar10` | CNN | 0.1500 | 75/500 | 14.3768 |
| `preresnet1001_cifar10` | CNN | 0.2000 | 100/500 | 19.1690 |
| `vit_hf_aaraki` | ViT | 0.1620 | 81/500 | 15.5269 |
| `vit_hf_nateraw` | ViT | 0.1320 | 66/500 | 12.6515 |
| `vit_timm_edadaltocg` | ViT | 0.1300 | 65/500 | 12.4599 |
| `robust_engstrom` | Robust | 0.2720 | 136/500 | 26.0698 |
| `robust_rade_r18_extra` | Robust | 0.2160 | 108/500 | 20.7025 |
| `robust_rebuffi_70_16_cutmix_extra` | Robust | 0.1460 | 73/500 | 13.9934 |
| `robust_xcit_s12` | Robust | 0.1860 | 93/500 | 17.8272 |
| `robust_xcit_m12` | Robust | 0.1560 | 78/500 | 14.9518 |
| `robust_sehwag_resnest152` | Robust | 0.1820 | 91/500 | 17.4438 |
