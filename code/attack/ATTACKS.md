# 对抗攻击算法说明

本项目针对 CIFAR-10（32×32）实现了 **5 个迁移攻击算法** + **1 个 SSIM 收缩后处理器**。

---

## 运行环境

```bash
conda activate causal
# 或通过 conda run：
conda run -n causal python workspace/code/attack/run_xxx.py
```

GPU：RTX 4070 Laptop（8 GB VRAM）。所有脚本均支持断点续跑（跳过 `results/adv_xxx/` 中已存在的文件）。

---

## 模型池

| 类型 | 数量 | 说明 |
|------|------|------|
| SURROGATE（代理模型，攻击可用） | 6 | `config.py::SURROGATE_NAMES` |
| HOLDOUT（裁判模型，攻击禁止接触） | 14 | `eval/models.py::HOLDOUT_MODELS` |

6 个代理模型（各架构族取一）：

```
pyramidnet164_a270_bn_cifar10
wrn16_10_cifar10
densenet190_k40_bc_cifar10
seresnet110_cifar10
resnext29_16x64d_cifar10
diaresnet56_cifar10
```

攻击超参：`eps=8/255`，`alpha=2/255`，`steps=10`，`batch_size=10`。

---

## 评分公式

```
Score = 100 × pool_ASR × mean_SSIM
```

- ASR 用 `workspace/data/label.txt` 真标签计算
- SSIM 对全部 500 张取均值
- **决策依据 holdout score**（surrogate ASR 可能虚高，holdout 是无偏代理）

---

## 算法列表

### 1. ILPD — Intermediate-Level Perturbation Decay

| 项 | 值 |
|---|---|
| 脚本 | `run_ilpd.py` |
| 输出 | `results/adv_ilpd/` |
| 论文 | ECCV 2022 |
| 仓库 | https://github.com/qizhangli/ILPD-attack |

**原理**：在模型中间层特征上施加扰动，通过衰减中间层特征偏离来引导 delta 更新。相比输入空间 PGD 有更强的迁移性。

**CIFAR-10 适配**：
- 主模型换为 `pyramidnet164_a270_bn_cifar10`（精度最高的代理）
- hook 挂在 `features.stage2`（CIFAR-10 pytorchcv 结构对应层）
- `coef=0.1`，去掉 ImageNet 数据加载逻辑

**特别说明**：ILPD 论文有明确 CIFAR-10 实验（+3.88% 迁移提升），是本项目中唯一在原论文中验证过 CIFAR-10 的算法，可信度最高。

---

### 2. PGN — Penalizing Gradient Norm

| 项 | 值 |
|---|---|
| 脚本 | `run_pgn.py` |
| 输出 | `results/adv_pgn/` |
| 论文 | ICCV 2023 |
| 仓库 | https://github.com/Trustworthy-AI-Group/PGN |

**原理**：在 MI-FGSM 基础上对梯度范数加惩罚项，使扰动落入更平坦的损失区域，提升迁移性（类似 SAM 但作用于输入而非权重）。

**CIFAR-10 适配**：
- 替换 InceptionV3 为 6 个 pytorchcv CIFAR-10 代理集成
- 去掉 ImageNet CSV 加载器，改用 PNG 批读取
- 原生实现，无外部仓库依赖

---

### 3. BSR — Block Shuffle and Rotation

| 项 | 值 |
|---|---|
| 脚本 | `run_bsr.py` |
| 输出 | `results/adv_bsr/` |
| 论文 | CVPR 2024 |
| 仓库 | https://github.com/Trustworthy-AI-Group/TransferAttack |

**原理**：将图像分块后随机打乱 + 旋转，作为输入变换增强（Input Transformation family），通过多次变换期望梯度提升迁移性。

**CIFAR-10 适配**：
- 使用 TransferAttack 库中的 BSR 实现
- `num_block` 适配 32×32（不超过图像宽度）
- 通过 `sys.path` 优先加载 TransferAttack repo，避免与 AWT 的 `transferattack` 包冲突

---

### 4. SIT — Structure Invariant Transformation / SIA

| 项 | 值 |
|---|---|
| 脚本 | `run_sit.py` |
| 输出 | `results/adv_sit/` |
| 论文 | ICCV 2023 |
| 仓库 | https://github.com/xiaosen-wang/SIT |
| 依赖 | `torch_dct==0.1.6`（`pip install torch-dct`） |

**原理**：在 DCT 域对图像块做结构不变变换（平移/旋转/缩放），构造 `num_copies` 份变换副本取平均梯度，提升对结构性特征的攻击迁移性。

**CIFAR-10 适配**：
- 直接将 pytorchcv ensemble 作为模型传入 SIA（接口兼容）
- `num_block=4`（32/8=4，适配 32×32）
- `num_copies=10`（原论文用 20，减半节省显存）
- 需安装 `torch_dct`（DCT 变换依赖）

---

### 5. AWT — Adversarial Weight Transfer

| 项 | 值 |
|---|---|
| 脚本 | `run_awt.py` |
| 输出 | `results/adv_awt/` |
| 论文 | NeurIPS 2023 |
| 仓库 | https://github.com/xaddwell/AWT |

**原理**：在每个攻击步骤内，用 SAM（Sharpness-Aware Minimization）扰动代理模型权重，寻找更锐利的损失区域，再在此处计算对 delta 的梯度。让攻击在"最坏情况的模型权重附近"优化，提升对未知模型的迁移性。

**CIFAR-10 适配**：
- `load_model()` 返回 pytorchcv CIFAR-10 ensemble（6 模型）
- SAM 只扰动**主模型**（第一个，`pyramidnet164_a270_bn`）的权重，其余 5 个冻结
  - 原因：SAM 扰动全部 6 个模型权重会 OOM（8 GB 不够）
  - 实现：`_EnsembleWithPrimaryParams` 覆盖 `parameters()` 只暴露主模型参数
- `run_utils.build_awt_ensemble()` 负责此差异化初始化

---

## SSIM 收缩后处理

```bash
python shrink_delta.py --adv-dir results/adv_pgn --out-dir results/adv_pgn_shrunk
```

**原理**：对每张已成功误分类的图像，二分搜索最小扰动缩放因子 `s ∈ [0,1]`，使 `clean + s×delta` 仍然误分类。减小 delta 幅度 → 提高 SSIM → 提高 Score。

**参数**：
- `--bisect 12`：二分搜索步数（默认 12，精度约 1/4096）
- `--min-scale 0.0`：允许缩至 0（若模型已对干净图像误分类）

**预期收益**：SSIM 提升约 0.02–0.05，Score 提升约 2–5 点（取决于原始算法 delta 幅度）。

---

## 评测与结果汇总

```bash
# 评测所有算法（需先完成各算法运行）
conda run -n causal python workspace/code/attack/run_eval_all.py
```

输出：
- `results/eval_<algo>.json`：各算法 surrogate + holdout ASR/SSIM/Score
- `results/adv_images/`：最优算法的图像（按 holdout score 选）
- `results/metrics.json`：最优算法摘要
- `artifacts/attack/results.json`：完整对比表

---

## 未采用算法

| 算法 | 放弃原因 |
|------|---------|
| **L2T**（Learning to Transform，CVPR 2024） | L2T 每步执行 search + attack 双 forward，batch 被 `num_scale` 扩展后激活张量峰值超出 8 GB VRAM，在 RTX 4070 Laptop 上无法运行。根本原因：L2T 的变换搜索空间在 224×224 设计，移植到 32×32 后计算图仍然过大，收益不确定但代价过高。 |

---

## 运行顺序建议

```bash
# 1. 生成各算法对抗样本（可并行）
conda run -n causal python workspace/code/attack/run_ilpd.py
conda run -n causal python workspace/code/attack/run_pgn.py
conda run -n causal python workspace/code/attack/run_bsr.py
conda run -n causal python workspace/code/attack/run_sit.py
conda run -n causal python workspace/code/attack/run_awt.py

# 2. 可选：对各算法结果做 SSIM 收缩
conda run -n causal python workspace/code/attack/shrink_delta.py \
    --adv-dir results/adv_pgn --out-dir results/adv_pgn_shrunk

# 3. 评测，选最优算法
conda run -n causal python workspace/code/attack/run_eval_all.py
```
