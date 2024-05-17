import torch

from problems import autogradient
from problems.exponential import exponential_gradient, exponential_hessian, exponential_function


def test_exponential_gradient():
    x = torch.tensor([-1.2, 2.6, -0.4, 1], dtype=torch.double)

    gradient_at_x = exponential_gradient(x)
    print(f"exponential_gradient(x)={gradient_at_x}")
    auto_gradient_at_x = autogradient.gradient(exponential_function, x)
    print(f"auto gradient(x)={auto_gradient_at_x}")

    norm_delta = torch.linalg.norm(gradient_at_x - auto_gradient_at_x, 2).item()
    assert norm_delta < 10e-1


def test_exponential_hessian():
    x = torch.tensor([-1.2, 2.6, -0.4, 1], dtype=torch.double)

    hessian_at_x = exponential_hessian(x)
    print(f"exponential_hessian(x)={hessian_at_x}")
    auto_hessian_at_x = autogradient.hessian(exponential_function, x)
    print(f"auto hessian(x)={auto_hessian_at_x}")

    norm_delta = torch.linalg.norm(hessian_at_x - auto_hessian_at_x, 2).item()
    assert norm_delta < 10e-1
