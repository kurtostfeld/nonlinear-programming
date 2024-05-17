from typing import Optional

import torch


def quadratic_function(big_q: torch.Tensor, small_q: torch.Tensor, x: torch.Tensor) -> float:
    return torch.dot(x, torch.matmul(big_q, x)).item() / 2 + torch.dot(small_q, x).item()


def quadratic_gradient(big_q: torch.Tensor, small_q: torch.Tensor, x: torch.Tensor) -> torch.tensor:
    return torch.matmul(big_q, x) + small_q


def quadratic_hessian(big_q: torch.Tensor, _1: torch.Tensor, _2: torch.Tensor) -> torch.tensor:
    return big_q
