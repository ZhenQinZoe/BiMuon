from dataclasses import dataclass
import torch

@dataclass
class DatasetBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    w_star: torch.Tensor
    theta_star: torch.Tensor
