import math
from typing import Optional, Callable, NamedTuple

import torch

from problems.problems import ProblemFunctionType, GradientFunctionType
from methods.methods import OptimizationOptions


class LineSearchState(NamedTuple):
    function_value: float
    gradient: torch.Tensor
    search_direction: torch.Tensor


type LineSearchFunctionType = Callable[[
    OptimizationOptions,
    ProblemFunctionType,
    GradientFunctionType,
    torch.Tensor,
    Optional[LineSearchState],
    LineSearchState], float]


def armijo_backtracking(options: OptimizationOptions,
                        f: ProblemFunctionType,
                        _: Callable[[torch.Tensor], torch.Tensor],
                        x: torch.Tensor,
                        previous: Optional[LineSearchState],
                        current: LineSearchState) -> float:
    searchk_dot_gradk = torch.dot(current.search_direction, current.gradient).item()
    if searchk_dot_gradk > 0:
        # this shouldn't happen.
        raise "error: searchk_dot_gradk is positive"

    alpha_init = 1
    if previous:
        phi_slope = torch.dot(previous.gradient, previous.search_direction).item()
        if phi_slope > 0:
            # this shouldn't happen.
            raise "error: phi_slope is positive"
        alpha_init = 2 * (current.function_value - previous.function_value) / phi_slope

    alpha = alpha_init
    for contraction_iter in range(1, options.armijo_backtracking_max_iterations):
        xkp1 = x + alpha * current.search_direction
        fkp1 = f(xkp1)

        expected_decrease = options.armijo_backtracking_c1 * alpha * searchk_dot_gradk
        expected_target = current.function_value + expected_decrease
        if fkp1 <= expected_target:
            return alpha

        alpha = alpha * options.armijo_backtracking_contraction_factor

    return 0


def wolfe_line_search(options: OptimizationOptions,
                      f: Callable[[torch.Tensor], float],
                      gradient_function: Callable[[torch.Tensor], torch.Tensor],
                      x: torch.Tensor,
                      _: Optional[LineSearchState],
                      current: LineSearchState) -> float:
    step_length_alpha_lower_limit = 0
    step_length_alpha_upper_limit = math.inf
    step_length_alpha = 1

    phi_prime_zero = torch.dot(current.search_direction, current.gradient).item()
    if phi_prime_zero > 0:
        # this shouldn't happen.
        raise "error: search_direction dot gradient is positive"

    for i in range(options.wolfe_line_search_max_iterations):
        x_after_step = x + step_length_alpha * current.search_direction
        phi_alpha = f(x_after_step)

        if phi_alpha > current.function_value + options.wolfe_line_search_c1 * step_length_alpha * phi_prime_zero:
            step_length_alpha_upper_limit = step_length_alpha
        else:
            gradient_after_step = gradient_function(x_after_step)
            phi_prime_alpha = torch.dot(current.search_direction, gradient_after_step).item()
            if phi_prime_alpha < options.wolfe_line_search_c2 * phi_prime_zero:
                step_length_alpha_lower_limit = step_length_alpha
            else:
                return step_length_alpha

        if not math.isinf(step_length_alpha_upper_limit):
            step_length_alpha = (step_length_alpha_lower_limit + step_length_alpha_upper_limit) / 2
        else:
            step_length_alpha = 2 * step_length_alpha

    # print("wolfe line search hit iteration limit")
    return 0
