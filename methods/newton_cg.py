import torch

from methods.methods import OptimizationOptions
from problems.problems import HessianFunctionType


def calc_newton_cg_search_direction(options: OptimizationOptions, xk: torch.Tensor, gradk: torch.Tensor,
                                    hessian_function: HessianFunctionType) -> torch.Tensor:
    gradk_norm = torch.linalg.norm(gradk, 2).item()
    eta_gradk_norm = options.newton_cg_eta_tolerance * gradk_norm
    hessk = hessian_function(xk)

    n = gradk.shape[0]
    zj = torch.zeros(n, dtype=torch.double)
    rj = gradk
    dj = -1 * rj

    for j in range(options.newton_cg_seaerch_direction_max_iterations):
        djt_hessk_dj = torch.dot(dj, torch.matmul(hessk, dj)).item()
        if djt_hessk_dj <= 0:
            if j == 0:
                return dj
            else:
                return zj

        rj_normsq = torch.dot(rj, rj).item()
        alphaj = rj_normsq / djt_hessk_dj

        # next
        zjp1 = zj + alphaj * dj
        rjp1 = rj + alphaj * torch.matmul(hessk, dj)
        rjp1_norm = torch.linalg.norm(rjp1, 2).item()

        if rjp1_norm <= eta_gradk_norm:
            return zjp1

        betajp1 = torch.dot(rjp1, rjp1).item() / torch.dot(rj, rj).item()
        djp1 = -1 * rjp1 + betajp1 * dj

        zj = zjp1
        rj = rjp1
        dj = djp1

    return torch.zeros(n, dtype=torch.double)
