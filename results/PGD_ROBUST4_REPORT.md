# PGD Robust4 方案简报

记录日期：2026-06-11  
参考文件：`workspace/results/SMOKE_RESULTS.md`

## 方案定义

本方案使用 `run_pgd.py` 的 plain IFGSM/PGD-style 攻击，不使用 DI/SI/TI、momentum、EOT 等迁移增强技巧。它的定位是 robust 模型的诊断基线：如果纯 PGD 在白盒 robust surrogate 上也难以推进，则更复杂的迁移增强方法大概率也不会直接解决 robust 攻击问题。

攻击池只包含 4 个 RobustBench / robust surrogate：

| 类别 | 模型 |
|---|---|
| Robust | `robust_engstrom` |
| Robust | `robust_rade_r18_extra` |
| Robust | `robust_rebuffi_70_16_cutmix_extra` |
| Robust | `robust_xcit_s12` |

不使用 CNN surrogate，不使用 ViT surrogate。

## 关键配置

| 参数 | 值 |
|---|---|
| runner | `workspace/code/attack/run_pgd.py` |
| attack type | Plain PGD / IFGSM |
| attack pool | `4 Robust` |
| loss | `crossentropy` |
| steps | `20` |
| epsilon | `8/255` |
| alpha | `2/255` |
| random start | off |
| smoke limit | first 8 images |
| smoke output | `workspace/results/adv_pgd_robust4_smoke20` |

对应命令形态：

```bash
python code/attack/run_pgd.py \
  --out-dir results/adv_pgd_robust4_smoke20 \
  --max-images 8 \
  --steps 20 \
  --batch-size 1 \
  --cnn-count 0 \
  --vit-surrogates none \
  --robust-surrogates robust_engstrom,robust_rade_r18_extra,robust_rebuffi_70_16_cutmix_extra,robust_xcit_s12 \
  --loss ce
```

## Smoke 结果

`SMOKE_RESULTS.md` 中记录的本地评测结果如下，均为前 8 张图像：

| Eval pool | Limit | SSIM | ASR | Score | 结论 |
|---|---:|---:|---:|---:|---|
| robust | 8 | 0.9517 | 0.1458 | 13.8787 | 当前最好的 robust smoke，但成功率仍低 |
| vit | 8 | 0.9517 | 0.1250 | 11.8960 | robust-only 对 ViT 迁移弱 |
| holdout | 8 | 0.9517 | 0.1250 | 11.8960 | robust-only 对 CNN holdout 迁移弱 |

这里的 Score 定义为：

```text
Score = 100 * pool_ASR * mean_SSIM
```

## 对比判断

从 `SMOKE_RESULTS.md` 的结论看：

| 观察 | 说明 |
|---|---|
| Mixed `2 CNN + 2 ViT + 4 Robust` 方法 robust ASR 多次为 0 | 混合平均梯度会稀释 robust 梯度 |
| `PGD robust4` 是目前 robust smoke 最好结果 | 说明 robust-only 梯度方向对 robust pool 有一定帮助 |
| `PGD robust4` 对 ViT/CNN 迁移很弱 | 它更像 robust 专项阶段，不适合作为唯一最终方案 |
| full run 需要重新干净运行 | 现有 `adv_pgd_robust4_full20` 只有 81 张且曾 OOM/中断，不应作为正式结果 |

## 当前状态

| 输出目录 | 状态 |
|---|---|
| `adv_pgd_robust4_smoke20` | smoke 完成，8 PNG |
| `adv_pgd_robust4_full20` | 非干净 partial full，81 PNG，不用于比较 |

## 建议

下一步建议重新开一个全新目录跑完整 500 张，例如：

```bash
python code/attack/run_pgd.py \
  --out-dir results/adv_pgd_robust4_full20_clean \
  --steps 20 \
  --batch-size 1 \
  --cnn-count 0 \
  --vit-surrogates none \
  --robust-surrogates robust_engstrom,robust_rade_r18_extra,robust_rebuffi_70_16_cutmix_extra,robust_xcit_s12 \
  --loss ce
```

如果显存确认稳定，可以再把 `batch-size` 提高到正常值；否则优先用 `batch-size=1` 保证完整性。完成后再评估 robust / vit / holdout 三个池，决定是否把它作为第一阶段，再接 ILPD 或 mixed diverse 阶段。
