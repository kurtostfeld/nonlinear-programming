import math
import torch


def genhumps_function(x: torch.Tensor) -> float:
    sum = 0
    for i in range(1, len(x)):
        xi = x[i-1].item()
        xip1 = x[i].item()

        sinxi = math.sin(2*xi)
        sinxip1 = math.sin(2*xip1)
        sum += (sinxi**2)*(sinxip1**2)
        sum += 0.05 * ((xi**2) + (xip1**2))

    return sum


def genhumps_gradient(x: torch.Tensor) -> torch.Tensor:
    x1 = x[0].item()
    x2 = x[1].item()
    x3 = x[2].item()
    x4 = x[3].item()
    x5 = x[4].item()

    grad = [
        0.1 * x1 + 4 * math.sin(2*x1) * ((math.sin(2*x2))**2) * math.cos(2*x1),
        0.2 * x2 + 4 * ((math.sin(2*x1))**2) * math.sin(2*x2) * math.cos(2*x2) + 4 * math.sin(2*x2) * ((math.sin(2*x3))**2) * math.cos(2*x2),
        0.2 * x3 + 4 * ((math.sin(2*x2))**2) * math.sin(2*x3) * math.cos(2*x3) + 4 * math.sin(2*x3) * ((math.sin(2*x4))**2) * math.cos(2*x3),
        0.2 * x4 + 4 * ((math.sin(2*x3))**2) * math.sin(2*x4) * math.cos(2*x4) + 4 * math.sin(2*x4) * ((math.sin(2*x5))**2) * math.cos(2*x4),
        0.1 * x5 + 4 * ((math.sin(2*x4))**2) * math.sin(2*x5) * math.cos(2*x5)
    ]

    return torch.tensor(grad, dtype=torch.double)


def genhumps_hessian(x: torch.Tensor) -> torch.Tensor:
    x1 = x[0].item()
    x2 = x[1].item()
    x3 = x[2].item()
    x4 = x[3].item()
    x5 = x[4].item()

    hess = [[0 for _ in range(5)] for _ in range(5)]
    hess[0][0] = -8 * ((math.sin(2*x1))**2) * ((math.sin(2*x2))**2) + 8 * ((math.sin(2*x2))**2) * ((math.cos(2*x1))**2) + 0.1
    hess[1][1] = -8 * ((math.sin(2*x1))**2) * ((math.sin(2*x2))**2) + 8 * ((math.sin(2*x1))**2) * ((math.cos(2*x2))**2) \
            - 8 * ((math.sin(2*x2))**2) * ((math.sin(2*x3))**2) + 8 * ((math.sin(2*x3))**2) * ((math.cos(2*x2))**2) + 0.2
    hess[2][2] = -8 * ((math.sin(2*x2))**2) * ((math.sin(2*x3))**2) + 8 * ((math.sin(2*x2))**2) * ((math.cos(2*x3))**2) \
            - 8 * ((math.sin(2*x3))**2) * ((math.sin(2*x4))**2) + 8 * ((math.sin(2*x4))**2) * ((math.cos(2*x3))**2) + 0.2
    hess[3][3] = -8 * ((math.sin(2*x3))**2) * ((math.sin(2*x4))**2) + 8 * ((math.sin(2*x3))**2) * ((math.cos(2*x4))**2) \
            - 8 * ((math.sin(2*x4))**2) * ((math.sin(2*x5))**2) + 8 * ((math.sin(2*x5))**2) * ((math.cos(2*x4))**2) + 0.2
    hess[4][4] = -8 * ((math.sin(2*x4))**2) * ((math.sin(2*x5))**2) + 8 * ((math.sin(2*x4))**2) * ((math.cos(2*x5))**2) + 0.1

    hess[0][1] = 2 * math.cos(4*x1 - 4*x2) - 2 * math.cos(4*x1 + 4*x2)
    hess[1][2] = 2 * math.cos(4*x2 - 4*x3) - 2 * math.cos(4*x2 + 4*x3)
    hess[2][3] = 2 * math.cos(4*x3 - 4*x4) - 2 * math.cos(4*x3 + 4*x4)
    hess[3][4] = 2 * math.cos(4*x4 - 4*x5) - 2 * math.cos(4*x4 + 4*x5)
    hess[1][0] = hess[0][1]
    hess[2][1] = hess[1][2]
    hess[3][2] = hess[2][3]
    hess[4][3] = hess[3][4]

    return torch.tensor(hess, dtype=torch.double)
