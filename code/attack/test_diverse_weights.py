import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_diverse import compute_shortfall_weights, untargeted_margin_loss


def test_shortfall_weights_focus_lower_asr_models():
    asr = torch.tensor([0.90, 0.30, 0.60])
    weights = compute_shortfall_weights(asr, floor=0.10, temperature=4.0)

    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert torch.all(weights >= 0.10)
    assert weights[1] > weights[2] > weights[0]


def test_shortfall_weights_do_not_drop_strong_models():
    asr = torch.tensor([1.0, 0.0, 1.0, 0.0])
    weights = compute_shortfall_weights(asr, floor=0.08, temperature=5.0)

    assert torch.all(weights >= 0.08)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_untargeted_margin_loss_increases_non_true_logit_margin():
    logits = torch.tensor([
        [5.0, 3.0, 1.0],
        [0.1, 2.0, 1.0],
    ])
    labels = torch.tensor([0, 1])

    loss = untargeted_margin_loss(logits, labels)

    assert torch.isclose(loss, torch.tensor(-1.5))


if __name__ == "__main__":
    test_shortfall_weights_focus_lower_asr_models()
    test_shortfall_weights_do_not_drop_strong_models()
    test_untargeted_margin_loss_increases_non_true_logit_margin()
    print("diverse weight tests passed")
