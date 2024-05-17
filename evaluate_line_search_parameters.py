import csv
import os

from generate_html_benchmarks import mkdir_generated
from methods.methods import Method, OptimizationOptions
from methods.run_optimization import timed_run_optimization
from problems.problems import load_problem, ProblemType, Problem


def try_c1_values(problem: Problem, method: Method):
    file_path = os.path.join("generated", "c1_experiment_data.csv")
    with open(file_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["c1", "step_index", "gradient_norm"])

        c1_values = [0.0006, 0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.0020, 0.0022, 0.0024]
        for c1 in c1_values:
            options = OptimizationOptions(wolfe_line_search_c1=c1)
            results = timed_run_optimization(method, problem, options)

            for step_index, step in enumerate(results.results.steps):
                csv_writer.writerow([c1, step_index, step.after_gradient_norm])


def try_c2_values(problem: Problem, method: Method):
    file_path = os.path.join("generated", "c2_experiment_data.csv")
    with open(file_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["c2", "step_index", "gradient_norm"])

        c2_values = [0.70, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96]
        for c2 in c2_values:
            options = OptimizationOptions(wolfe_line_search_c2=c2)
            results = timed_run_optimization(method, problem, options)

            for step_index, step in enumerate(results.results.steps):
                csv_writer.writerow([c2, step_index, step.after_gradient_norm])


def go():
    mkdir_generated()

    problem = load_problem(ProblemType.Genhumps_5)
    method = Method.ModifiedNewtonW

    try_c1_values(problem, method)
    try_c2_values(problem, method)


if __name__ == '__main__':
    go()
