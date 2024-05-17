from typing import Callable

import torch


def _partial_derivative(f: Callable[[torch.Tensor], float], x0: torch.Tensor, i: int) -> float:
    epsilon = 1e-5
    x1 = torch.clone(x0)
    x1[i] += epsilon

    f0 = f(x0)
    f1 = f(x1)
    return (f1 - f0) / epsilon


def _partial_derivative_2(f: Callable[[torch.Tensor], float], x0: torch.Tensor, i: int, j: int) -> float:
    def partial1(z: torch.Tensor) -> float:
        return _partial_derivative(f, z, i)

    return _partial_derivative(partial1, x0, j)


def gradient(f: Callable[[torch.Tensor], float], x0: torch.Tensor) -> torch.Tensor:
    grad = [_partial_derivative(f, x0, i) for i in range(x0.shape[0])]
    return torch.tensor(grad, dtype=torch.double)


def hessian(f: Callable[[torch.Tensor], float], x0: torch.Tensor) -> torch.Tensor:
    hess = [[_partial_derivative_2(f, x0, i, j) for j in range(x0.shape[0])] for i in range(x0.shape[0])]
    return torch.tensor(hess, dtype=torch.double)
