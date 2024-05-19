README
======

# Introduction #

This is a final project that I did as a student for course ORI 391.Q nonlinear programming that I took in Spring 2024 with [Professor Bollapragada](https://sites.google.com/view/raghub/home). The [assignment](docs/assignment.pdf) involved implementing several different optimization algorithms on several different problems and comparing success minimization metrics and convergence properties.

Browse the Python+PyTorch source code here. Generated results can be seen [here](https://kurtostfeld.github.io/nonlinear-programming/)

# Instructions to Run Code #

To run this code, first setup and active a Python virtual environment. All the other instructions in this README will assume that such an environment has been setup and activated.

# virtualenv setup #

```bash
cd nonlinear-programming
rm -rf venv
# This is assuming you are using Homebrew. If you are using some other environment, adjust this command accordingly.
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv venv
source ./venv/bin/activate
pip3 install --upgrade pip setuptools
pip3 install -r requirements.txt
```

# Run code lint and unit tests #

Run `flake8` lint code verification and `pytest` unit tests:

```bash
make flake8
make test
```

# See the function that matches the final project assignment #

The final project assignment specifies a function of the form:

```
[x, f] = optSolver-TeamName(problem,method,options)
```

That function exists in `final_project.py` with some example usage.

# Generate tabular results in .html format #

```bash
# optionally remove any old results.
rm -rf generated/

python generate_html_benchmarks.py

# open .html files in generated/ directory.
```

# Generate line search parameter comparison results in .csv format #

```bash
# optionally remove any old results.
rm -rf generated/

python evaluate_line_search_parameters.py
```
