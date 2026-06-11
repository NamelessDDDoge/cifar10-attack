# Model Inventory for PGD Robust4 Evaluation

This table covers all models mentioned in the full-500 evaluation summary: CNN, ViT, and Robust.

| Category | Evaluation name | Concrete architecture | Weight source |
|---|---|---|---|
| CNN | `pyramidnet164_a270_bn_cifar10` | PyramidNet-164, alpha=270, BN, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\pyramidnet164_a270_bn_cifar10-0242-daa2a402.pth` |
| CNN | `wrn16_10_cifar10` | WideResNet-16-10, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\wrn16_10_cifar10-0293-ce810d8a.pth` |
| CNN | `densenet190_k40_bc_cifar10` | DenseNet-190, growth=40, bottleneck+compression, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\densenet190_k40_bc_cifar10-0252-2896fa08.pth` |
| CNN | `seresnet110_cifar10` | SE-ResNet-110, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\seresnet110_cifar10-0363-1ddec230.pth` |
| CNN | `resnext29_16x64d_cifar10` | ResNeXt-29, 16x64d, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\resnext29_16x64d_cifar10-0241-4133d3d0.pth` |
| CNN | `diaresnet56_cifar10` | DIA-ResNet-56, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\diaresnet56_cifar10-0505-8ac86804.pth` |
| CNN | `pyramidnet236_a220_bn_cifar10` | PyramidNet-236, alpha=220, BN, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\pyramidnet236_a220_bn_cifar10-0247-daa91d74.pth` |
| CNN | `pyramidnet110_a270_cifar10` | PyramidNet-110, alpha=270, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\pyramidnet110_a270_cifar10-0251-31bdd9d5.pth` |
| CNN | `pyramidnet272_a200_bn_cifar10` | PyramidNet-272, alpha=200, BN, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\pyramidnet272_a200_bn_cifar10-0239-586b1ecd.pth` |
| CNN | `densenet250_k24_bc_cifar10` | DenseNet-250, growth=24, bottleneck+compression, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\densenet250_k24_bc_cifar10-0267-f8f9d305.pth` |
| CNN | `pyramidnet110_a84_cifar10` | PyramidNet-110, alpha=84, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\pyramidnet110_a84_cifar10-0298-7b835a3c.pth` |
| CNN | `wrn20_10_1bit_cifar10` | 1-bit WideResNet-20-10, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\wrn20_10_1bit_cifar10-0326-e6140f8a.pth` |
| CNN | `wrn28_10_cifar10` | WideResNet-28-10, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\wrn28_10_cifar10-0239-fe97dcd6.pth` |
| CNN | `resnet272bn_cifar10` | ResNet-272 with BN, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\resnet272bn_cifar10-0333-84f28e0c.pth` |
| CNN | `pyramidnet200_a240_bn_cifar10` | PyramidNet-200, alpha=240, BN, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\pyramidnet200_a240_bn_cifar10-0244-44433afd.pth` |
| CNN | `resnext272_1x64d_cifar10` | ResNeXt-272, 1x64d, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\resnext272_1x64d_cifar10-0255-070ccc35.pth` |
| CNN | `seresnet164bn_cifar10` | SE-ResNet-164 with BN, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\seresnet164bn_cifar10-0339-1085dab6.pth` |
| CNN | `wrn20_10_32bit_cifar10` | 32-bit WideResNet-20-10, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\wrn20_10_32bit_cifar10-0314-a18146e8.pth` |
| CNN | `wrn40_8_cifar10` | WideResNet-40-8, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\wrn40_8_cifar10-0237-8dc84ec7.pth` |
| CNN | `preresnet1001_cifar10` | Pre-activation ResNet-1001, CIFAR-10 | pytorchcv pretrained CIFAR-10; local cache `C:\Users\admin\.torch\models\preresnet1001_cifar10-0265-9fedfe5f.pth` |
| ViT | `vit_hf_aaraki` | ViT-B/16, 224px, ImageNet-21k pretrain, CIFAR-10 finetune | HuggingFace `aaraki/vit-base-patch16-224-in21k-finetuned-cifar10`; local cache `C:\Users\admin\.cache\huggingface\hub\models--aaraki--vit-base-patch16-224-in21k-finetuned-cifar10\...` |
| ViT | `vit_hf_nateraw` | ViT-B/16, 224px, CIFAR-10 finetune | HuggingFace `nateraw/vit-base-patch16-224-cifar10`; local cache `C:\Users\admin\.cache\huggingface\hub\models--nateraw--vit-base-patch16-224-cifar10\...` |
| ViT | `vit_timm_edadaltocg` | timm ViT-B/16, 224px, ImageNet-21k pretrain, CIFAR-10 finetune | HuggingFace/timm `edadaltocg/vit_base_patch16_224_in21k_ft_cifar10`; local cache `C:\Users\admin\.cache\huggingface\hub\models--edadaltocg--vit_base_patch16_224_in21k_ft_cifar10\...` |
| Robust | `robust_engstrom` | RobustBench `Engstrom2019Robustness`; ResNet bottleneck layout `[3,4,6,3]` (ResNet-50 style), CIFAR-10 Linf | RobustBench model zoo; local file `C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\models\cifar10\Linf\Engstrom2019Robustness.pt` |
| Robust | `robust_rade_r18_extra` | RobustBench `Rade2021Helper_R18_extra`; DMPreActResNet-18, extra data, CIFAR-10 Linf | RobustBench model zoo; local file `C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\models\cifar10\Linf\Rade2021Helper_R18_extra.pt` |
| Robust | `robust_rebuffi_70_16_cutmix_extra` | RobustBench `Rebuffi2021Fixing_70_16_cutmix_extra`; DM WideResNet-70-16, Swish, CutMix + extra data, CIFAR-10 Linf | RobustBench model zoo; local file `C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\models\cifar10\Linf\Rebuffi2021Fixing_70_16_cutmix_extra.pt` |
| Robust | `robust_xcit_s12` | RobustBench `Debenedetti2022Light_XCiT-S12`; XCiT-S12, CIFAR-10 Linf | timm/RobustBench; local checkpoint `C:\Users\admin\.cache\torch\hub\checkpoints\debenedetti2022light-xcit-s-cifar10-linf.pth.tar` |
| Robust | `robust_xcit_m12` | RobustBench `Debenedetti2022Light_XCiT-M12`; XCiT-M12, CIFAR-10 Linf | timm/RobustBench; local checkpoint `C:\Users\admin\.cache\torch\hub\checkpoints\debenedetti2022light-xcit-m-cifar10-linf.pth.tar` |
| Robust | `robust_sehwag_resnest152` | RobustBench `Sehwag2021Proxy_ResNest152`; ResNeSt-152, CIFAR-10 Linf | RobustBench model zoo; local file `C:\文件\ME\AISCI\Dr.Researcher\projects\aisafety-cifar10-attack\models\cifar10\Linf\Sehwag2021Proxy_ResNest152.pt` |
