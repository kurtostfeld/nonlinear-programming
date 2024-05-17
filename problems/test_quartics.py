import pytest

from problems.problems import load_problem, ProblemType


def test_quartics():
    problem = load_problem(ProblemType.P5_quartic_1)
    assert problem.objective_function(problem.x0) == pytest.approx(1.0019, 1e-2)

    problem = load_problem(ProblemType.P6_quartic_2)
    assert problem.objective_function(problem.x0) == pytest.approx(1.9196e+05, 1e+1)
