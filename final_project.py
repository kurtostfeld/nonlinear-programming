import torch

from methods.methods import OptimizationOptions, Method
from methods.run_optimization import run_optimization
from problems.problems import Problem, ProblemType, load_problem


# This is the function structure specified by the final project.
#
# [x, f] = optSolver-TeamName(problem,method,options)
def team_kurt_ostfeld(problem: Problem, method: Method, options: OptimizationOptions) -> (torch.tensor, float):
    optimization_results = run_optimization(method, problem, options)
    return optimization_results.final_x, optimization_results.final_function_value


if __name__ == "__main__":
    # Example usage.
    # Choose and load a problem:
    problem = load_problem(ProblemType.P1_quad_10_10)
    print(f"{problem.problem_type}: f(x0)={problem.objective_function(problem.x0)}")
    # Choose a method
    method = Method.ModifiedNewtonW
    # Choose options. There are sensible defaults for everything.
    # Here we are specifying one custom option.
    options = OptimizationOptions(wolfe_line_search_c2=0.8)

    # run.
    final_x, final_f = team_kurt_ostfeld(problem, Method.ModifiedNewtonW, options)

    # print results
    print(f"done. final_x={final_x}, final_function_value={final_f}")
