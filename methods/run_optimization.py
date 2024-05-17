import datetime
import time

from methods.bfgs_optimization import run_l_bfgs_optimization_loop, \
    run_quasi_newton_optimization_loop, bfgs_update, dfp_update
from methods.line_search import armijo_backtracking, wolfe_line_search
from methods.newton_cg import calc_newton_cg_search_direction
from methods.optimization_loop_simple import run_optimization_loop_simple, calc_search_direction_steepest_descent, \
    calc_search_direction_newton_modified
from methods.methods import Method, OptimizationOptions, OptimizationResults, TimedOptimizationResults
from problems.problems import Problem


def timed_run_optimization(method: Method, problem: Problem, options: OptimizationOptions) -> TimedOptimizationResults:
    # timedelta
    start_datetime = datetime.datetime.now()
    results = run_optimization(method, problem, options)
    run_time = datetime.datetime.now() - start_datetime
    return TimedOptimizationResults(results, run_time)


def run_optimization(method: Method, problem: Problem, options: OptimizationOptions) -> OptimizationResults:
    match method:
        case Method.GradientDescent:
            return run_optimization_loop_simple(
                calc_search_direction_steepest_descent, armijo_backtracking, problem, options)
        case Method.GradientDescentW:
            return run_optimization_loop_simple(
                calc_search_direction_steepest_descent, wolfe_line_search, problem, options)
        case Method.ModifiedNewton:
            return run_optimization_loop_simple(
                calc_search_direction_newton_modified, armijo_backtracking, problem, options)
        case Method.ModifiedNewtonW:
            return run_optimization_loop_simple(
                calc_search_direction_newton_modified, wolfe_line_search, problem, options)
        case Method.NewtonCG:
            return run_optimization_loop_simple(
                calc_newton_cg_search_direction, armijo_backtracking, problem, options)
        case Method.NewtonCGW:
            return run_optimization_loop_simple(
                calc_newton_cg_search_direction, wolfe_line_search, problem, options)
        case Method.BFGS:
            return run_quasi_newton_optimization_loop(
                armijo_backtracking, bfgs_update, problem, options)
        case Method.BFGSW:
            return run_quasi_newton_optimization_loop(
                wolfe_line_search, bfgs_update, problem, options)
        case Method.DFP:
            return run_quasi_newton_optimization_loop(
                armijo_backtracking, dfp_update, problem, options)
        case Method.DFPW:
            return run_quasi_newton_optimization_loop(
                wolfe_line_search, dfp_update, problem, options)
        case Method.LBFGS:
            return run_l_bfgs_optimization_loop(
                armijo_backtracking, problem, options)
        case Method.LBFGSW:
            return run_l_bfgs_optimization_loop(
                wolfe_line_search, problem, options)
        case _:
            raise NotImplementedError(f"unknown method type: {method}")
