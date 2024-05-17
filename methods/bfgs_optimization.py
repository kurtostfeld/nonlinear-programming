from typing import Callable

import math

import torch

from methods.line_search import LineSearchState, LineSearchFunctionType
from methods.methods import OptimizationOptions, OptimizationResults, OptimizationStep, OptimizationTerminationReason
from problems.problems import Problem


type QuasiNewtonUpdateCallable = Callable[[
    OptimizationOptions, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def dfp_update(options: OptimizationOptions, _: torch.Tensor,
               hk: torch.Tensor, sk: torch.Tensor, yk: torch.Tensor) -> torch.Tensor:
    yk_dot_sk = torch.dot(yk, sk).item()
    yk_norm = torch.linalg.norm(yk, 2).item()
    sk_norm = torch.linalg.norm(sk, 2).item()

    if yk_dot_sk <= options.bfgs_update_epsilon_min * yk_norm * sk_norm:
        return hk

    sk_outer = torch.outer(sk, sk)
    yk_outer = torch.outer(yk, yk)

    first_denom = torch.dot(yk, torch.matmul(hk, yk)).item()
    first_num = torch.matmul(torch.matmul(hk, yk_outer), hk)
    first = first_num / first_denom

    second = sk_outer / yk_dot_sk

    return hk - first + second


def bfgs_update(options: OptimizationOptions, identity_n: torch.Tensor,
                hk: torch.Tensor, sk: torch.Tensor, yk: torch.Tensor) -> torch.Tensor:
    yk_dot_sk = torch.dot(yk, sk).item()
    yk_norm = torch.linalg.norm(yk, 2).item()
    sk_norm = torch.linalg.norm(sk, 2).item()

    if yk_dot_sk <= options.bfgs_update_epsilon_min * yk_norm * sk_norm:
        return hk

    sk_outer_yk = torch.outer(sk, yk)
    yk_outer_sk = torch.outer(yk, sk)
    # sk_outer_yk.transpose()

    left = identity_n - sk_outer_yk / yk_dot_sk
    right = identity_n - yk_outer_sk / yk_dot_sk
    first = torch.matmul(torch.matmul(left, hk), right)

    sk_outer = torch.outer(sk, sk)
    second = sk_outer / yk_dot_sk
    return first + second


def run_quasi_newton_optimization_loop(line_search_function: LineSearchFunctionType,
                                       hk_quasi_newton_update: QuasiNewtonUpdateCallable,
                                       problem: Problem, options: OptimizationOptions) -> OptimizationResults:
    xk = problem.x0
    fk = problem.objective_function(xk)
    gradk = problem.objective_gradient_function(xk)
    gradk_norm = torch.linalg.norm(gradk, 2).item()

    identity_n = torch.diag(torch.ones(problem.x0.shape[0], dtype=torch.double))
    # Use identity for H_0 or the initial inverse hessian.
    hk = torch.diag(torch.ones(problem.x0.shape[0], dtype=torch.double))

    step_list = []
    step = OptimizationStep(0, fk, gradk_norm)
    step_list.append(step)
    options.step_callback(0, step)

    previous_line_search_state = None

    for i in range(1, options.max_iterations + 1):
        searchk = -1 * torch.matmul(hk, gradk)
        current_line_search_state = LineSearchState(fk, gradk, searchk)

        step_length_alpha = line_search_function(options, problem.objective_function,
                                                 problem.objective_gradient_function,
                                                 xk, previous_line_search_state, current_line_search_state)

        # set previous
        previous_line_search_state = current_line_search_state
        fkm1 = fk
        gradkm1 = gradk
        gradkm1_norm = gradk_norm

        # take step. update
        sk = step_length_alpha * searchk
        xk = xk + sk
        fk = problem.objective_function(xk)
        gradk = problem.objective_gradient_function(xk)
        gradk_norm = torch.linalg.norm(gradk, 2).item()
        yk = gradk - gradkm1

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

        hk = hk_quasi_newton_update(options, identity_n, hk, sk, yk)

    return OptimizationResults(step_list, xk, fk, gradk_norm,
                               OptimizationTerminationReason.reached_iteration_limit)


def l_bfgs_get_gammak(sks_and_yks: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    if len(sks_and_yks) > 0:
        sk, yk = sks_and_yks[0]
        num = torch.dot(sk, yk).item()
        denom = torch.dot(yk, yk).item()
        return num / denom
    else:
        return 1


def l_bfgs_recursion(identity: torch.Tensor, gradk: torch.Tensor,
                     sks_and_yks: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    gammak = l_bfgs_get_gammak(sks_and_yks)
    hk = gammak * identity

    q = gradk
    alphaks = []
    for sk_and_yk in sks_and_yks:
        sk, yk = sk_and_yk
        rhok = 1 / torch.dot(yk, sk).item()
        alphak = rhok * torch.dot(sk, q).item()
        alphaks.append(alphak)
        q = q - alphak * yk

    r = torch.matmul(hk, q)

    for (sk_and_yk, alphak) in zip(reversed(sks_and_yks), reversed(alphaks)):
        sk, yk = sk_and_yk
        rhok = 1 / torch.dot(yk, sk).item()
        beta = rhok * torch.dot(yk, r).item()
        r = r + sk * (alphak - beta)

    return -1 * r


def run_l_bfgs_optimization_loop(line_search_function: LineSearchFunctionType,
                                 problem: Problem, options: OptimizationOptions) -> OptimizationResults:
    n = problem.x0.shape[0]
    m = min([10, n])
    identity = torch.diag(torch.ones(n, dtype=torch.double))
    sks_and_yks = []

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
        searchk = l_bfgs_recursion(identity, gradk, sks_and_yks)
        current_line_search_state = LineSearchState(fk, gradk, searchk)

        step_length_alpha = line_search_function(options, problem.objective_function,
                                                 problem.objective_gradient_function,
                                                 xk, previous_line_search_state, current_line_search_state)

        # set previous
        previous_line_search_state = current_line_search_state
        fkm1 = fk
        gradkm1_norm = gradk_norm

        # take step. update
        sk = step_length_alpha * searchk
        xk = xk + sk
        fk = problem.objective_function(xk)
        gradk = problem.objective_gradient_function(xk)
        yk = gradk - current_line_search_state.gradient
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

        yk_dot_sk = torch.dot(yk, sk).item()
        yk_norm = torch.linalg.norm(yk, 2).item()
        sk_norm = torch.linalg.norm(sk, 2).item()

        if yk_dot_sk > options.l_bfgs_update_epsilon_min * yk_norm * sk_norm:
            sks_and_yks.insert(0, (sk, yk))
            if len(sks_and_yks) > m:
                sks_and_yks.pop()

    return OptimizationResults(step_list, xk, fk, gradk_norm,
                               OptimizationTerminationReason.reached_iteration_limit)
