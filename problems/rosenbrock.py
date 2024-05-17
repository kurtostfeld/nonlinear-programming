import torch


def rosenbrock_function(x: torch.Tensor) -> float:
    n = x.shape[0]

    f = 0
    # variable i is 1-indexed to match document.
    for i in range(1, n):
        # adjust 1-based indices to 0-based python indices.
        xi = x[i-1].item()
        xip1 = x[i].item()

        fi = 100 * (xip1 - xi**2)**2 + (1 - xi)**2
        f = f + fi
    return f


def rosenbrock_gradient(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]

    grad = [0] * n

    x1 = x[0].item()
    x2 = x[1].item()
    grad[0] = -400 * x1 * x2 + 400 * x1**3 - 2 + 2*x1

    for i in range(2, n):
        xim1 = x[i - 2].item()
        xi = x[i - 1].item()
        xip1 = x[i].item()
        grad[i-1] = 400 * xi**3 + 202*xi - 200*xim1**2 - 400*xi*xip1 - 2

    xnm1 = x[n - 2].item()
    xn = x[n - 1].item()
    grad[n - 1] = 200 * xn - 200 * xnm1**2

    return torch.tensor(grad, dtype=torch.double)


def rosenbrock_hessian(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]

    hess = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(1, n+1):
        if i == 1:
            x1 = x[0].item()
            x2 = x[1].item()
            hess[i-1][i-1] = -400 * x2 + 1200 * x1**2 + 2
            hess[i-1][i] = -400 * x1
            hess[i][i-1] = hess[i-1][i]
        elif i == n:
            xnm1 = x[n - 2].item()
            hess[i-1][i-1] = 200
            hess[i-1][i-2] = -400*xnm1
            hess[i-2][i-1] = hess[i-1][i-2]
        else:
            xim1 = x[i - 2].item()
            xi = x[i - 1].item()
            xip1 = x[i].item()
            hess[i-1][i-1] = 1200 * xi**2 + 202 - 400 * xip1
            hess[i-1][i-2] = -400 * xim1
            hess[i-2][i-1] = hess[i-1][i-2]
            hess[i-1][i] = -400*xi
            hess[i][i-1] = hess[i-1][i]

    return torch.tensor(hess, dtype=torch.double)
