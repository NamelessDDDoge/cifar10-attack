import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ilpd import project_initial_images


def test_project_initial_images_clamps_to_clean_epsilon_ball():
    clean = torch.full((1, 3, 2, 2), 0.5)
    init = torch.tensor([[[[0.0, 1.0], [0.53, 0.47]]] * 3])

    projected = project_initial_images(clean, init, epsilon=0.1)

    assert torch.all(projected >= 0.4)
    assert torch.all(projected <= 0.6)
    assert torch.all(projected >= 0.0)
    assert torch.all(projected <= 1.0)
