import torch

from problems import autogradient
from problems.rosenbrock import rosenbrock_gradient, rosenbrock_function, rosenbrock_hessian


def test_rosenbrock_gradient():
    x = torch.tensor([-1.2, 2.6, -0.4, 1], dtype=torch.double)

    gradient_at_x = rosenbrock_gradient(x)
    print(f"rosenbrock_gradient(x)={gradient_at_x}")
    auto_gradient_at_x = autogradient.gradient(rosenbrock_function, x)
    print(f"auto gradient(x)={auto_gradient_at_x}")

    norm_delta = torch.linalg.norm(gradient_at_x - auto_gradient_at_x, 2).item()
    assert norm_delta < 10e-1


def test_rosenbrock_hessian():
    x = torch.tensor([-1.2, 2.6, -0.4, 1], dtype=torch.double)

    hessian_at_x = rosenbrock_hessian(x)
    print(f"rosenbrock_hessian(x)={hessian_at_x}")
    auto_hessian_at_x = autogradient.hessian(rosenbrock_function, x)
    print(f"auto hessian(x)={auto_hessian_at_x}")

    norm_delta = torch.linalg.norm(hessian_at_x - auto_hessian_at_x, 2).item()
    assert norm_delta < 10e-1
