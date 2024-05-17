from math import cos, sin

import torch


def build_q() -> torch.Tensor:
    return torch.tensor([[5,1,0,0.5], [1,4,0.5,0], [0,0.5,3,0], [0.5,0,0,2]], dtype=torch.double)


def build_x0() -> torch.Tensor:
    return torch.tensor([cos(70), sin(70), cos(70), sin(70)], dtype=torch.double)


def quartic_function(q: torch.tensor, sigma: float, x: torch.Tensor) -> float:
    return sigma * torch.dot(x, torch.matmul(q, x)).item() / 4 + torch.dot(x, x).item() / 2


def quartic_gradient(q: torch.tensor, sigma: float, x: torch.Tensor) -> torch.Tensor:
    return sigma * torch.matmul(q, x) / 2 + x


def quartic_hessian(q: torch.tensor, _1: float, _2: torch.Tensor) -> torch.Tensor:
    return q
