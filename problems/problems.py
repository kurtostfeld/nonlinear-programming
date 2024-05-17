import functools
import os
from enum import StrEnum, auto
from typing import NamedTuple, Callable, Optional

import torch

from problems import quadratics, quartics
from problems.beale import beale_hessian, beale_gradient, beale_function
from problems.exponential import exponential_function, exponential_gradient, exponential_hessian
from problems.genhumps import genhumps_function, genhumps_gradient, genhumps_hessian
from problems.load_csv import load_csv_to_tensor
from problems.rosenbrock import rosenbrock_function, rosenbrock_gradient, rosenbrock_hessian

type ProblemFunctionType = Callable[[torch.Tensor], float]
type GradientFunctionType = Callable[[torch.Tensor], torch.Tensor]
type HessianFunctionType = Callable[[torch.Tensor], torch.Tensor]

class ProblemType(StrEnum):
    P1_quad_10_10 = auto()
    P2_quad_10_1000 = auto()
    P3_quad_1000_10 = auto()
    P4_quad_1000_1000 = auto()
    P5_quartic_1 = auto()
    P6_quartic_2 = auto()
    Rosenbrock_2 = auto()
    Rosenbrock_100 = auto()
    DataFit_2 = auto()
    Exponential_10 = auto()
    Exponential_1000 = auto()
    Genhumps_5 = auto()


class Problem(NamedTuple):
    problem_type: ProblemType
    objective_function: ProblemFunctionType
    objective_gradient_function: GradientFunctionType
    objective_hessian_function: HessianFunctionType
    x0: torch.Tensor


def load_problem(problem_type: ProblemType) -> Problem:
    objective_function: Optional[ProblemFunctionType]
    objective_gradient_function: Optional[GradientFunctionType]
    objective_hessian_function: Optional[HessianFunctionType]
    x0: Optional[torch.Tensor]

    match problem_type:
        case ProblemType.P1_quad_10_10:
            small_q = load_csv_to_tensor(os.path.join("data", "small-q-10-10.txt"))
            big_q = load_csv_to_tensor(os.path.join("data", "big-q-10-10.txt"))
            x0 = load_csv_to_tensor(os.path.join("data", "x0-10.txt"))

            objective_function = functools.partial(quadratics.quadratic_function, big_q, small_q)
            objective_gradient_function = functools.partial(quadratics.quadratic_gradient, big_q, small_q)
            objective_hessian_function = functools.partial(quadratics.quadratic_hessian, big_q, small_q)
        case ProblemType.P2_quad_10_1000:
            small_q = load_csv_to_tensor(os.path.join("data", "small-q-10-1000.txt"))
            big_q = load_csv_to_tensor(os.path.join("data", "big-q-10-1000.txt"))
            x0 = load_csv_to_tensor(os.path.join("data", "x0-10.txt"))

            objective_function = functools.partial(quadratics.quadratic_function, big_q, small_q)
            objective_gradient_function = functools.partial(quadratics.quadratic_gradient, big_q, small_q)
            objective_hessian_function = functools.partial(quadratics.quadratic_hessian, big_q, small_q)
        case ProblemType.P3_quad_1000_10:
            small_q = load_csv_to_tensor(os.path.join("data", "small-q-1000-10.txt"))
            big_q = load_csv_to_tensor(os.path.join("data", "big-q-1000-10.txt"))
            x0 = load_csv_to_tensor(os.path.join("data", "x0-1000.txt"))

            objective_function = functools.partial(quadratics.quadratic_function, big_q, small_q)
            objective_gradient_function = functools.partial(quadratics.quadratic_gradient, big_q, small_q)
            objective_hessian_function = functools.partial(quadratics.quadratic_hessian, big_q, small_q)
        case ProblemType.P4_quad_1000_1000:
            small_q = load_csv_to_tensor(os.path.join("data", "small-q-1000-1000.txt"))
            big_q = load_csv_to_tensor(os.path.join("data", "big-q-1000-1000.txt"))
            x0 = load_csv_to_tensor(os.path.join("data", "x0-1000.txt"))

            objective_function = functools.partial(quadratics.quadratic_function, big_q, small_q)
            objective_gradient_function = functools.partial(quadratics.quadratic_gradient, big_q, small_q)
            objective_hessian_function = functools.partial(quadratics.quadratic_hessian, big_q, small_q)
        case ProblemType.P5_quartic_1:
            q = quartics.build_q()
            x0 = quartics.build_x0()
            sigma = 1e-4

            objective_function = functools.partial(quartics.quartic_function, q, sigma)
            objective_gradient_function = functools.partial(quartics.quartic_gradient, q, sigma)
            objective_hessian_function = functools.partial(quartics.quartic_hessian, q, sigma)
        case ProblemType.P6_quartic_2:
            q = quartics.build_q()
            x0 = quartics.build_x0()
            sigma = 1e4

            objective_function = functools.partial(quartics.quartic_function, q, sigma)
            objective_gradient_function = functools.partial(quartics.quartic_gradient, q, sigma)
            objective_hessian_function = functools.partial(quartics.quartic_hessian, q, sigma)
        case ProblemType.Rosenbrock_2:
            x0 = torch.tensor([-1.2, 1], dtype=torch.double)

            objective_function = rosenbrock_function
            objective_gradient_function = rosenbrock_gradient
            objective_hessian_function = rosenbrock_hessian
        case ProblemType.Rosenbrock_100:
            x0 = torch.full([100], 1, dtype=torch.double)
            x0[0] = -1.2

            objective_function = rosenbrock_function
            objective_gradient_function = rosenbrock_gradient
            objective_hessian_function = rosenbrock_hessian
        case ProblemType.DataFit_2:
            x0 = torch.tensor([1, 1], dtype=torch.double)

            objective_function = beale_function
            objective_gradient_function = beale_gradient
            objective_hessian_function = beale_hessian
        case ProblemType.Exponential_10:
            raw = [0] * 10
            raw[0] = 1
            x0 = torch.tensor(raw, dtype=torch.double)

            objective_function = exponential_function
            objective_gradient_function = exponential_gradient
            objective_hessian_function = exponential_hessian
        case ProblemType.Exponential_1000:
            raw = [0] * 100
            raw[0] = 1
            x0 = torch.tensor(raw, dtype=torch.double)

            objective_function = exponential_function
            objective_gradient_function = exponential_gradient
            objective_hessian_function = exponential_hessian
        case ProblemType.Genhumps_5:
            x0 = torch.tensor([-506.2, 506.2, 506.2, 506.2, 506.2], dtype=torch.double)

            objective_function = genhumps_function
            objective_gradient_function = genhumps_gradient
            objective_hessian_function = genhumps_hessian
        case _:
            raise NotImplementedError(f"unknown problem type: {problem_type}")

    return Problem(
        problem_type,
        objective_function,
        objective_gradient_function,
        objective_hessian_function,
        x0)
