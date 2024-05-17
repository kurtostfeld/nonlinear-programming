import math
import torch


def exponential_function(x: torch.Tensor) -> float:
    x1 = x[0]
    exp_x1 = math.exp(x1)
    sum = (exp_x1 - 1) / (exp_x1 + 1) + 0.1 * math.exp(-1 * x1)

    for i in range(2, len(x)):
        xi = x[i-1].item()
        sum += (xi - 1)**4

    return sum


def exponential_gradient(x: torch.Tensor) -> torch.Tensor:
    x1 = x[0]
    first_num = (1.9 * math.exp(2*x1) - 0.2 * math.exp(x1) - 0.1) * math.exp(-1 * x1)
    first_denom = math.exp(2*x1) + 2 * math.exp(x1) + 1.0

    grad = [0 for _ in range(len(x))]
    grad[0] = first_num / first_denom

    for i in range(2, len(x)):
        xi = x[i-1].item()
        grad[i-1] = 4 * (xi - 1)**3

    return torch.tensor(grad, dtype=torch.double)


def exponential_hessian(x: torch.Tensor) -> torch.Tensor:
    n = len(x)
    x1 = x[0]
    first_num = (-1.9 * math.exp(3*x1) + 2.3 * math.exp(2*x1) + 0.3 * math.exp(x1) + 0.1) * math.exp(-1*x1)
    first_denom = math.exp(3*x1) + 3 * math.exp(2*x1) + 3 * math.exp(x1) + 1

    hess = [[0 for _ in range(n)] for _ in range(n)]
    hess[0][0] = first_num / first_denom

    for i in range(2, n):
        xi = x[i-1].item()
        hess[i-1][i-1] = 12 * (xi - 1)**2

    return torch.tensor(hess, dtype=torch.double)
