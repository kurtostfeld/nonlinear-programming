import pytest

from problems.problems import load_problem, ProblemType


def test_quadratics():
    problem = load_problem(ProblemType.P1_quad_10_10)
    assert problem.objective_function(problem.x0) == pytest.approx(221.0900, 1e-2)

    problem = load_problem(ProblemType.P2_quad_10_1000)
    assert problem.objective_function(problem.x0) == pytest.approx(176.1591, 1e-2)

    problem = load_problem(ProblemType.P3_quad_1000_10)
    assert problem.objective_function(problem.x0) == pytest.approx(6.5508e+03, 1e+1)

    problem = load_problem(ProblemType.P4_quad_1000_1000)
    assert problem.objective_function(problem.x0) == pytest.approx(2.5468e+03, 1e+1)
