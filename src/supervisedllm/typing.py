from typing import Dict, List

import torch

Prediction = Dict[str, torch.Tensor]
Predictions = List[Prediction]
Target = Dict[str, torch.Tensor]
Targets = List[Dict[str, torch.Tensor]]
