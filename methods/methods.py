import datetime
from enum import StrEnum, auto
from typing import NamedTuple, Callable

import torch


class OptimizationTerminationReason(StrEnum):
    reached_stationary_point = auto()
    reached_iteration_limit = auto()
    function_and_gradient_norm_stopped_decreasing = auto()


class OptimizationStep(NamedTuple):
    step_length: float
    after_function_value: float
    after_gradient_norm: float


def step_callback_noop(k: int, step: OptimizationStep):
    pass


class OptimizationOptions(NamedTuple):
    max_iterations: int = 200
    step_callback: Callable[[int, OptimizationStep],None] = step_callback_noop
    stationary_point_gradient_norm_tolerance: float = 10e-5
    no_decrease_tolerance: float = 10e-8
    armijo_backtracking_contraction_factor: float = 0.5
    armijo_backtracking_c1: float = 10e-4
    armijo_backtracking_max_iterations: int = 50
    wolfe_line_search_max_iterations: int = 50
    wolfe_line_search_c1: float = 10e-4
    wolfe_line_search_c2: float = 0.9
    bfgs_update_epsilon_min: float = 1e-8
    l_bfgs_update_epsilon_min: float = 1e-8
    newton_cg_eta_tolerance: float = 0.01
    newton_cg_seaerch_direction_max_iterations: int = 50


class OptimizationResults(NamedTuple):
    steps: list[OptimizationStep]
    final_x: torch.Tensor
    final_function_value: float
    final_gradient_norm: float
    termination_reason: OptimizationTerminationReason


class TimedOptimizationResults(NamedTuple):
    results: OptimizationResults
    run_time: datetime.timedelta


class Method(StrEnum):
    GradientDescent = auto()
    GradientDescentW = auto()
    ModifiedNewton = auto()
    ModifiedNewtonW = auto()
    NewtonCG = auto()
    NewtonCGW = auto()
    BFGS = auto()
    BFGSW = auto()
    DFP = auto()
    DFPW = auto()
    LBFGS = auto()
    LBFGSW = auto()
