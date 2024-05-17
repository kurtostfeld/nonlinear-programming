import os

from methods.methods import Method, OptimizationOptions
from methods.run_optimization import timed_run_optimization
from problems.problems import ProblemType, load_problem
from methods.timedelta_format import timedelta_format


def mkdir_generated():
    try:
        os.mkdir("generated")
    except FileExistsError:
        # already exists. ok
        pass


def go():
    mkdir_generated()

    options = OptimizationOptions()
    for problem_type in ProblemType:
        problem = load_problem(ProblemType(problem_type))
        print(f"loaded. problem={problem_type}")

        file_path = os.path.join("generated", f"{problem_type}.html")
        with open(file_path, 'w') as html_file:
            html_file.write("<html><body>")
            html_file.write("<div>")
            html_file.write("<table>")
            html_file.write("<tr>")
            html_file.write("<th>method</th>")
            html_file.write("<th>final f</th>")
            html_file.write("<th>gradient norm</th>")
            html_file.write("<th>steps</th>")
            html_file.write("<th>time</th>")
            html_file.write("<th>termination reason</th>")
            html_file.write("</tr>")

            for method in Method:
                print(f"method={method}. problem={problem_type}")
                results = timed_run_optimization(Method(method), problem, options)
                html_file.write("<tr>")
                html_file.write(f"<td>{method.name}</td>")
                html_file.write(f"<td>{results.results.final_function_value:<14.5f}</td>")
                html_file.write(f"<td>{results.results.final_gradient_norm:<10.5f}</td>")
                html_file.write(f"<td>{len(results.results.steps)}</td>")
                html_file.write(f"<td>{timedelta_format(results.run_time)}</td>")
                html_file.write(f"<td>{results.results.termination_reason}</td>")
                html_file.write("</tr>")

            html_file.write("</table>")
            html_file.write("</div>")
            html_file.write("</body></html>")


if __name__ == '__main__':
    go()
