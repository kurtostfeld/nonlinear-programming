import torch

from problems import autogradient
from problems.beale import beale_gradient, beale_function, beale_hessian


def test_beale_gradient():
    x = torch.tensor([1.5, -0.4], dtype=torch.double)

    gradient_at_x = beale_gradient(x)
    print(f"grad_beale(x)={gradient_at_x}")
    auto_gradient_at_x = autogradient.gradient(beale_function, x)
    print(f"auto gradient(x)={auto_gradient_at_x}")

    norm_delta = torch.linalg.norm(gradient_at_x - auto_gradient_at_x, 2).item()
    assert norm_delta < 10e-1


def test_beale_hessian():
    x = torch.tensor([1.5, -0.4], dtype=torch.double)

    hessian_at_x = beale_hessian(x)
    print(f"grad_beale(x)={hessian_at_x}")
    auto_hessian_at_x = autogradient.hessian(beale_function, x)
    print(f"auto gradient(x)={auto_hessian_at_x}")

    norm_delta = torch.linalg.norm(hessian_at_x - auto_hessian_at_x, 2).item()
    assert norm_delta < 10e-1
