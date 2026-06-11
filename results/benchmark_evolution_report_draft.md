# CIFAR-10 迁移攻击 Benchmark 与 Surrogate 选择策略演化报告草稿

记录日期：2026-06-11  
数据来源：`workspace/results/external_clean_accuracy_summary.json`、`workspace/results/SMOKE_RESULTS.md`、`workspace/results/eval_*.json`、`artifacts/experiment/*/summary.md`。  
计分方式：除特别说明外，攻击评分采用 `Score = 100 * pool_ASR * mean_SSIM`。其中 ASR 表示模型或模型池被对抗样本攻击成功的比例，SSIM 用于约束扰动的感知相似度。

## 1. 实验 Benchmark 的初始建立：以 clean accuracy 选择高分模型

最初的 benchmark 构建遵循了一个直观假设：若一个预训练 CIFAR-10 模型在干净样本上的准确率更高，则它更可能代表“更强”的评测目标；相应地，能在这些高准确率模型上获得更高攻击分数的对抗样本，也应当具有更强的迁移攻击能力。因此，第一版 benchmark 直接对可加载的预训练权重进行 clean accuracy 评测，并按 500 张评测图像上的准确率排序，选取前 20 个模型作为主要评测对象。

这一策略的优点是客观、可复现，且实现成本低；但它隐含地将 clean accuracy 等同于迁移攻击难度或迁移代表性。后续实验表明，这一假设并不充分：高 clean accuracy 模型大量集中在普通 CNN 系列，无法覆盖 ViT 与 adversarially robust 模型的不同决策边界。

| Rank | Model | Family | Loader | Clean acc (500) | Correct / 500 |
|---:|---|---|---|---:|---:|
| 1 | `vit_hf_aaraki` | ViT | `hf_transformers` | 0.994 | 497 |
| 2 | `vit_hf_nateraw` | ViT | `hf_transformers` | 0.992 | 496 |
| 3 | `vit_timm_edadaltocg` | ViT | `timm_hf` | 0.990 | 495 |
| 4 | `robust_rebuffi_70_16_cutmix_extra` | Robust | `robustbench` | 0.976 | 488 |
| 5 | `chenyaofo_mobilenetv2_x0_75` | CNN | `torch_hub` | 0.976 | 488 |
| 6 | `chenyaofo_repvgg_a0` | CNN | `torch_hub` | 0.974 | 487 |
| 7 | `chenyaofo_repvgg_a2` | CNN | `torch_hub` | 0.974 | 487 |
| 8 | `chenyaofo_mobilenetv2_x1_0` | CNN | `torch_hub` | 0.972 | 486 |
| 9 | `robust_xcit_s12` | Robust | `robustbench` | 0.970 | 485 |
| 10 | `chenyaofo_resnet56` | CNN | `torch_hub` | 0.968 | 484 |
| 11 | `chenyaofo_repvgg_a1` | CNN | `torch_hub` | 0.968 | 484 |
| 12 | `robust_xcit_m12` | Robust | `robustbench` | 0.964 | 482 |
| 13 | `chenyaofo_resnet32` | CNN | `torch_hub` | 0.964 | 482 |
| 14 | `chenyaofo_mobilenetv2_x0_5` | CNN | `torch_hub` | 0.964 | 482 |
| 15 | `chenyaofo_resnet44` | CNN | `torch_hub` | 0.962 | 481 |
| 16 | `chenyaofo_shufflenetv2_x1_0` | CNN | `torch_hub` | 0.962 | 481 |
| 17 | `chenyaofo_shufflenetv2_x1_5` | CNN | `torch_hub` | 0.962 | 481 |
| 18 | `chenyaofo_vgg16_bn` | CNN | `torch_hub` | 0.960 | 480 |
| 19 | `chenyaofo_shufflenetv2_x2_0` | CNN | `torch_hub` | 0.960 | 480 |
| 20 | `chenyaofo_resnet20` | CNN | `torch_hub` | 0.958 | 479 |

该表显示，若只按 clean accuracy 排序，benchmark 很容易被高准确率的常规模型主导。虽然前三名恰好是 ViT，但 Top-20 中普通 CNN 占据多数；与此同时，robust 模型虽然进入列表，但数量与结构多样性不足，无法单独反映 robust transfer 的难度。

## 2. Benchmark 修正：从“高分模型”转向“按模型家族覆盖”

进一步评测后发现，clean accuracy 排名前列的模型并不能保证对迁移攻击形成充分约束。尤其是在 RobustBench / adversarially robust 模型上，许多能有效攻击普通 CNN 或 ViT 的样本几乎没有攻击成功率。因此，benchmark 被修正为按模型家族显式划分：普通 CNN、ViT、Robust 三类都必须进入评测池，并分别报告模型池 ASR 与分数。

这种划分的关键作用是将“平均表现”拆解为结构化表现。若只报告一个合并分数，普通 CNN 上的高 ASR 会掩盖 robust 模型上的失败；按家族评测则能暴露迁移攻击在不同模型类型上的短板。

| Family | # Models | Mean clean acc | Min clean acc | Max clean acc | Included models |
|---|---:|---:|---:|---:|---|
| ViT | 3 | 0.992 | 0.990 | 0.994 | `vit_hf_aaraki`, `vit_hf_nateraw`, `vit_timm_edadaltocg` |
| Robust | 6 | 0.957 | 0.930 | 0.976 | `robust_engstrom`, `robust_xcit_s12`, `robust_xcit_m12`, `robust_rebuffi_70_16_cutmix_extra`, `robust_rade_r18_extra`, `robust_sehwag_resnest152` |
| CNN / chenyaofo | 19 | 0.963 | 0.946 | 0.976 | `chenyaofo_resnet20`, `chenyaofo_resnet32`, `chenyaofo_resnet44`, `chenyaofo_resnet56`, `chenyaofo_vgg11_bn`, `chenyaofo_vgg13_bn`, `chenyaofo_vgg16_bn`, `chenyaofo_vgg19_bn`, `chenyaofo_mobilenetv2_x0_5`, `chenyaofo_mobilenetv2_x0_75`, `chenyaofo_mobilenetv2_x1_0`, `chenyaofo_mobilenetv2_x1_4`, `chenyaofo_shufflenetv2_x0_5`, `chenyaofo_shufflenetv2_x1_0`, `chenyaofo_shufflenetv2_x1_5`, `chenyaofo_shufflenetv2_x2_0`, `chenyaofo_repvgg_a0`, `chenyaofo_repvgg_a1`, `chenyaofo_repvgg_a2` |

按家族重构 benchmark 后，实验目标从“攻击 clean accuracy 最高的若干模型”转变为“同时考察 CNN、ViT 与 robust 模型上的迁移性”。这一变化也改变了后续 surrogate pool 的设计逻辑：surrogate 不能只覆盖高分模型，而必须覆盖可能导致迁移失败的模型家族。

## 3. 攻击优化阶段：从 mixed surrogate 到 robust-only surrogate

这里更准确的术语不是“训练模型”，而是“攻击优化”或“对抗样本生成”。实验过程中并未重新训练目标模型参数，而是在固定 surrogate 模型池上，通过 PGD / IFGSM、ILPD 或 diverse-gradient 等方法对输入扰动进行迭代优化。

最初的攻击优化策略也遵循了 benchmark 的家族覆盖思想：既然评测池包含 CNN、ViT 与 Robust，那么 surrogate pool 也应包含每一类模型。典型配置为 `2 CNN + 2 ViT + 4 Robust`。但 smoke 实验显示，mixed surrogate 的平均梯度容易被普通 CNN / ViT 目标主导，robust 模型上的攻击成功率长期为 0。即使引入 ILPD 或 diverse shortfall-aware weighting，robust pool 仍然没有被有效提升。

| Run | Surrogate / method | Eval pool | Images | SSIM | ASR | Score | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| `adv_diverse_mixed_smoke20_r4` | Diverse, `2 CNN + 2 ViT + 4 Robust`, steps=20 | ViT | 8 | 0.9350 | 0.7083 | 66.2272 | ViT 迁移最强，但未解决 robust |
| `adv_diverse_mixed_smoke20_r4` | Diverse, `2 CNN + 2 ViT + 4 Robust`, steps=20 | Robust | 8 | 0.9350 | 0.0000 | 0.0000 | robust 完全失败 |
| `adv_diverse_mixed_smoke20_r4` | Diverse, `2 CNN + 2 ViT + 4 Robust`, steps=20 | CNN holdout | 8 | 0.9350 | 0.4911 | 45.9138 | 对普通 CNN 有迁移 |
| `adv_pgd_mixed_smoke20_r4` | Plain PGD / IFGSM, `2 CNN + 2 ViT + 4 Robust`, steps=20 | ViT | 8 | 0.9591 | 0.3750 | 35.9656 | ViT 有一定迁移 |
| `adv_pgd_mixed_smoke20_r4` | Plain PGD / IFGSM, `2 CNN + 2 ViT + 4 Robust`, steps=20 | Robust | 8 | 0.9591 | 0.0000 | 0.0000 | robust 仍为 0 |
| `adv_pgd_mixed_smoke20_r4` | Plain PGD / IFGSM, `2 CNN + 2 ViT + 4 Robust`, steps=20 | CNN holdout | 8 | 0.9591 | 0.6250 | 59.9427 | 普通 CNN 迁移较强 |
| `adv_ilpd_mixed_smoke20_r4` | ILPD, `2 CNN + 2 ViT + 4 Robust`, steps=20 | ViT | 8 | 0.9382 | 0.3750 | 35.1813 | 与 mixed PGD 接近 |
| `adv_ilpd_mixed_smoke20_r4` | ILPD, `2 CNN + 2 ViT + 4 Robust`, steps=20 | Robust | 8 | 0.9382 | 0.0000 | 0.0000 | robust 仍为 0 |
| `adv_ilpd_mixed_smoke20_r4` | ILPD, `2 CNN + 2 ViT + 4 Robust`, steps=20 | Surrogate CNN | 8 | 0.9382 | 0.7917 | 74.2717 | 白盒/近白盒 CNN 有效 |
| `adv_ilpd_mixed_smoke20_r4` | ILPD, `2 CNN + 2 ViT + 4 Robust`, steps=20 | CNN holdout | 8 | 0.9382 | 0.7500 | 70.3626 | CNN holdout 最强，但 robust 短板不变 |
| `adv_pgd_robust4_smoke20` | Plain PGD / IFGSM, `4 Robust`, steps=20 | Robust | 8 | 0.9517 | 0.1458 | 13.8787 | robust-only 首次显著提升 robust |
| `adv_pgd_robust4_smoke20` | Plain PGD / IFGSM, `4 Robust`, steps=20 | ViT | 8 | 0.9517 | 0.1250 | 11.8960 | ViT 迁移偏弱 |
| `adv_pgd_robust4_smoke20` | Plain PGD / IFGSM, `4 Robust`, steps=20 | CNN holdout | 8 | 0.9517 | 0.1250 | 11.8960 | CNN holdout 迁移偏弱 |

这组结果支持一个更具体的结论：简单增加 robust surrogate 的比例并不足以解决 robust pool 的低分问题；当 CNN、ViT 与 robust 梯度被直接平均时，攻击优化方向仍可能偏向更容易被攻击的非 robust 模型。将 surrogate pool 收缩为 robust-only 后，robust 评测池的 ASR 才开始提升。

## 4. Robust-only 完整运行与分阶段样本生成

在 smoke 结果基础上，进一步运行了 `adv_pgd_robust4_full20_clean`，使用 4 个 robust surrogate 进行 500 张图像的完整攻击优化。该实验不再使用 CNN 或 ViT surrogate，而是将 robust 模型作为主要梯度来源。完整 500 图评测显示，robust-only 的确提升了 robust pool 上的攻击成功率，但同时也说明其对 CNN/ViT 的泛化并非“全面强”，而是中等偏低。因此，当前更稳妥的表述是：robust-only 是修复 robust 短板的有效方向，但若追求总分，仍可能需要分阶段或组合式策略。

| Run | Attack source | Eval pool | Images | # Eval models | SSIM | ASR | Score | Evidence file |
|---|---|---|---:|---:|---:|---:|---:|---|
| `adv_pgd_robust4_full20_clean` | `4 Robust` | Robust | 500 | 4 | 0.9585 | 0.2050 | 19.6482 | `eval_pgd_robust4_full20_clean_full500_vit_robust.json` |
| `adv_pgd_robust4_full20_clean` | `4 Robust` | ViT | 500 | 2 | 0.9585 | 0.1310 | 12.5557 | `eval_pgd_robust4_full20_clean_full500_vit_robust.json` |
| `adv_pgd_robust4_full20_clean` | `4 Robust` | Surrogate CNN | 500 | 6 | 0.9585 | 0.1713 | 16.4214 | `eval_pgd_robust4_full20_clean_full500_surrogate_holdout.json` |
| `adv_pgd_robust4_full20_clean` | `4 Robust` | CNN holdout | 500 | 14 | 0.9585 | 0.1734 | 16.6223 | `eval_pgd_robust4_full20_clean_full500_surrogate_holdout.json` |
| `adv_stage_pgdrobust4_ilpd224_full20` | staged robust-PGD + ILPD-style route | ViT | 50 | 3 | 0.9462 | 0.4467 | 42.2643 | `eval_stage_pgdrobust4_ilpd224_full20_first50_vit_cpu.json` |
| `adv_stage_pgdrobust4_ilpd224_full20` | staged robust-PGD + ILPD-style route | Robust | 50 | 6 | 0.9462 | 0.1400 | 13.2470 | `eval_stage_pgdrobust4_ilpd224_full20_first50_robust_cpu.json` |
| `adv_stage_pgdrobust4_ilpd224_full20` | staged robust-PGD + ILPD-style route | CNN / chenyaofo | 50 | 19 | 0.9462 | 0.4484 | 42.4303 | `eval_stage_pgdrobust4_ilpd224_full20_first50_chenyaofo_cpu.json` |
| `adv_stage_pgdrobust4_ilpd224_full20` | staged robust-PGD + ILPD-style route | Surrogate CNN | 50 | 6 | 0.9462 | 0.7467 | 70.6507 | `eval_stage_pgdrobust4_ilpd224_full20_first50_all_cpu.json` |
| `adv_stage_pgdrobust4_ilpd224_full20` | staged robust-PGD + ILPD-style route | CNN holdout | 50 | 14 | 0.9462 | 0.7357 | 69.6144 | `eval_stage_pgdrobust4_ilpd224_full20_first50_all_cpu.json` |

完整 500 图 robust-only 结果的 per-model breakdown 如下。可以看到，不同 robust 模型之间的攻击难度仍然不同：`robust_engstrom` 的 ASR 最高，而 `robust_rebuffi_70_16_cutmix_extra` 最低。

| Model | Group | ASR | Attacked / Total |
|---|---|---:|---:|
| `vit_hf_nateraw` | ViT | 0.132 | 66 / 500 |
| `vit_timm_edadaltocg` | ViT | 0.130 | 65 / 500 |
| `robust_engstrom` | Robust | 0.272 | 136 / 500 |
| `robust_rade_r18_extra` | Robust | 0.216 | 108 / 500 |
| `robust_rebuffi_70_16_cutmix_extra` | Robust | 0.146 | 73 / 500 |
| `robust_xcit_s12` | Robust | 0.186 | 93 / 500 |

这一阶段的主要发现是：robust-only surrogate 能够把优化压力集中到 robust 决策边界上，从而避免 mixed surrogate 中 robust 梯度被稀释的问题；但是，robust-only 本身并不自动带来最强的 CNN/ViT 迁移。50 图 staged 结果显示，分阶段或组合式方法可能在 CNN/ViT 上获得更高 ASR，但 robust 仍然难以同步提升。这说明最终方案可能需要显式地在 robust transfer 与 CNN/ViT transfer 之间做多目标权衡，而不是依赖单一平均梯度。

## 5. 小结与当前结论

本轮实验的认识演化可以概括为三点。

第一，clean accuracy 不能直接作为迁移攻击 benchmark 的唯一筛选依据。Top-20 clean accuracy 模型虽然提供了一个可复现起点，但它会弱化模型家族差异，尤其不能充分暴露 robust 模型上的迁移失败。

第二，benchmark 必须按模型家族分层报告。CNN、ViT 与 Robust 模型的攻击难度和梯度结构不同，合并分数会掩盖重要失败模式。后续报告应固定输出每个家族的 pool ASR、SSIM、Score 与 per-model ASR。

第三，surrogate pool 的“多样性”并不等价于攻击有效性。直接混合 CNN、ViT 与 robust surrogate 在普通 CNN / ViT 上可以取得较高分数，但 robust ASR 多次为 0。robust-only surrogate 能显著改善 robust pool，但会牺牲部分 CNN/ViT 迁移。当前最合理的方向不是回到单一高准确率模型选择，而是采用分家族 benchmark，并继续探索 staged 或多目标优化策略。

## 6. 后续实验建议

1. 将 `adv_stage_pgdrobust4_ilpd224_full20` 从 50 图扩展到 500 图，验证其在 CNN/ViT 上的高 ASR 是否稳定。
2. 对 robust-only 与 staged 方法做同一 500 图、同一 eval pool 的可比实验，避免 smoke 结果与 full-run 结果混用。
3. 在报告主表中同时列出 `Robust score`、`ViT score`、`CNN holdout score`，并避免只使用一个合并平均分作为最终选择标准。
4. 若继续优化 mixed surrogate，应尝试显式的 robust loss reweighting 或分阶段扰动合成，而不是简单增加 robust 模型数量。
