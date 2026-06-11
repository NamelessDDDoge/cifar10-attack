<h1 align="center">Enhancing Adversarial Transferability with Adversarial Weight Tuning (AWT)</h1>


[(AAAI 2025) Enhancing Adversarial Transferability with Adversarial Weight Tuning](https://arxiv.org/abs/2408.09469):
AWT is a data-free tuning method that combines gradient-based and model-based attack methods to enhance the transferability of AEs, and proposes a new adversarial attack algorithm, which adaptively adjusts the parameters of the surrogate model using generated AEs to optimize the flat local maxima and model smoothness simultaneously, without the need for extra data. For gradient-based parts, AWT leverages the idea from [PGN (Ge et al., 2023)](https://arxiv.org/abs/2306.05225) which penalizes gradient norm on the original loss function. For model-based parts, AWT adopts the idea from [SAM](https://arxiv.org/abs/2010.01412) which achieve the flat landscape of the surrogate model.




## Usage

### Requirements
+ Python >= 3.6
+ PyTorch >= 1.12.1
+ Torchvision >= 0.13.1
+ timm >= 0.6.12

```bash
pip install -r requirements.txt
```

### Benign Dataset

First, you need to download the data from [![GoogleDrive](https://img.shields.io/badge/GoogleDrive-space-blue)
](https://drive.google.com/file/d/1d-_PKYi3MBDPtJV4rfMCCtmsE0oWX7ZB/view?usp=sharing) or [![Huggingface Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/datasets/Trustworthy-AI-Group/TransferAttack/blob/main/data.zip) into `/path/to/data`. Then you can execute the attack as follows:


### Run the attack
```
# generate adversarial samples
python main_awt.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --attack awt --model resnet50
# evaluate the transferability
python main_awt.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet18 --attack awt --eval
```

## Welcom to discuss with me
If you have any questions, please feel free to contact me at xaddwell@zju.edu.cn. I am focusing on Trustworthy AI (GenAI Security & Privacy, Distributed Learning Security & Privacy, etc).


## Acknowledgement
We thank all the researchers who contribute to the development of Transferable Adversarial Attack. Especially, we thank the benchmark [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack), provided by [Trustworthy-AI-Group](https://github.com/Trustworthy-AI-Group) and the authors of [PGN (Ge et al., 2023)](https://arxiv.org/abs/2306.05225) for their great work.

