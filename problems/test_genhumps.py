import torch

from problems import autogradient
from problems.genhumps import genhumps_gradient, genhumps_hessian, genhumps_function


def test_genhumps_gradient():
    x = torch.tensor([-506.2, 506.2, 506.2, 506.2, 506.2], dtype=torch.double)

    gradient_at_x = genhumps_gradient(x)
    print(f"genhumps_gradient(x)={gradient_at_x}")
    auto_gradient_at_x = autogradient.gradient(genhumps_function, x)
    print(f"auto gradient(x)={auto_gradient_at_x}")

    norm_delta = torch.linalg.norm(gradient_at_x - auto_gradient_at_x, 2).item()
    assert norm_delta < 10e-1


def test_genhumps_hessian():
    x = torch.tensor([-506.2, 506.2, 506.2, 506.2, 506.2], dtype=torch.double)

    hessian_at_x = genhumps_hessian(x)
    print(f"genhumps_hessian(x)={hessian_at_x}")
    auto_hessian_at_x = autogradient.hessian(genhumps_function, x)
    print(f"auto hessian(x)={auto_hessian_at_x}")

    norm_delta = torch.linalg.norm(hessian_at_x - auto_hessian_at_x, 2).item()
    assert norm_delta < 10e-1
