import math
from typing import Callable

import torch

from methods.line_search import LineSearchState, LineSearchFunctionType
from methods.methods import OptimizationOptions, OptimizationResults, OptimizationStep, OptimizationTerminationReason
from problems.problems import Problem, HessianFunctionType


def is_matrix_spd(a: torch.Tensor) -> bool:
    # Is A symmetric?
    if torch.linalg.norm(a - torch.t(a), 2).item() > 10e-4:
        return False

    # check for all positive eigenvalues.
    eig_val, _ = torch.linalg.eig(a)
    return torch.min(eig_val.float() > 0).item()


def modify_hessian(hess: torch.Tensor) -> torch.Tensor:
    beta = 10e-4

    n = torch.diagonal(hess).size(0)
    identity = torch.diag(torch.full([n], 1, dtype=torch.double))

    minaii = torch.min(torch.diagonal(hess))
    tau0 = 0 if minaii > 0 else beta - minaii

    tauk = tau0
    max_iterations = 50
    for k in range(max_iterations):
        modified_hess = hess + tauk * identity
        if is_matrix_spd(modified_hess):
            return modified_hess

        tauk = torch.max(torch.tensor([2*tauk, beta], dtype=torch.double)).item()

    raise torch.linalg.LinAlgError


# xk, gradient, HessianFunctionType
type SearchDirectionCalculationFunctionType = Callable[[OptimizationOptions, torch.Tensor, torch.Tensor, HessianFunctionType], torch.Tensor]

def calc_search_direction_steepest_descent(_options: OptimizationOptions, _xk: torch.Tensor, gradient: torch.Tensor,
                                           _hessian_function: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    return -1 * gradient


def calc_search_direction_newton_unmodified(_options: OptimizationOptions, xk: torch.Tensor, gradient: torch.Tensor,
                                            hessian_function: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    hessk = hessian_function(xk)
    searchk = -1 * torch.linalg.solve(hessk, gradient)
    return searchk


def calc_search_direction_newton_modified(_options: OptimizationOptions, xk: torch.Tensor, gradient: torch.Tensor,
                                          hessian_function: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    hessk = hessian_function(xk)
    modified_hessk = modify_hessian(hessk)
    searchk = -1 * torch.linalg.solve(modified_hessk, gradient)
    return searchk


def run_optimization_loop_simple(calc_search_direction: SearchDirectionCalculationFunctionType,
                                 line_search_function: LineSearchFunctionType,
                                 problem: Problem, options: OptimizationOptions) -> OptimizationResults:
    xk = problem.x0
    fk = problem.objective_function(xk)
    gradk = problem.objective_gradient_function(xk)
    gradk_norm = torch.linalg.norm(gradk, 2).item()

    step_list = []
    step = OptimizationStep(0, fk, gradk_norm)
    step_list.append(step)
    options.step_callback(0, step)

    previous_line_search_state = None

    for i in range(1, options.max_iterations + 1):
        searchk = calc_search_direction(options, xk, gradk, problem.objective_hessian_function)
        current_line_search_state = LineSearchState(fk, gradk, searchk)

        step_length_alpha = line_search_function(options, problem.objective_function,
                                                 problem.objective_gradient_function,
                                                 xk, previous_line_search_state, current_line_search_state)

        # set previous
        previous_line_search_state = current_line_search_state
        fkm1 = fk
        gradkm1_norm = gradk_norm

        # take step. update
        xk = xk + step_length_alpha * searchk
        fk = problem.objective_function(xk)
        gradk = problem.objective_gradient_function(xk)
        gradk_norm = torch.linalg.norm(gradk, 2).item()

        step = OptimizationStep(step_length_alpha, fk, gradk_norm)
        step_list.append(step)
        options.step_callback(i, step)

        if math.fabs(fkm1 - fk) < options.no_decrease_tolerance and \
                math.fabs(gradkm1_norm - gradk_norm) < options.no_decrease_tolerance:
            return OptimizationResults(step_list, xk, fk, gradk_norm,
                                       OptimizationTerminationReason.function_and_gradient_norm_stopped_decreasing)
        if gradk_norm < options.stationary_point_gradient_norm_tolerance:
            return OptimizationResults(step_list, xk, fk, gradk_norm,
                                       OptimizationTerminationReason.reached_stationary_point)

    return OptimizationResults(step_list, xk, fk, gradk_norm,
                               OptimizationTerminationReason.reached_iteration_limit)
