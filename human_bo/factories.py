import itertools
import torch
from botorch.acquisition import qMaxValueEntropy
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound


def pick_acqf(acqf, data, gpr, bounds):
    "Instantiate the given acqf."

    if acqf == "UCB":
        beta = 0.2  # basic value for normalized data
        af = UpperConfidenceBound(gpr, beta)
    elif acqf == "MES":
        Ncandids = 100  # size of candidate set to approximate MES
        candidate_set = torch.rand(
            Ncandids, bounds.size(1), device=bounds.device, dtype=bounds.dtype
        )
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        af = qMaxValueEntropy(gpr, candidate_set)
    else:
        af = ExpectedImprovement(gpr, data["train_Y"].max())
    return af


def build_combinations(N_REP, experiments, kernels, acqfs, n_init, seed):
    """Construct the list of combination settings to run."""

    combi = []
    li = [experiments, kernels, acqfs, [n_init], [seed + n for n in range(N_REP)]]
    combi.append(list(itertools.product(*li)))
    return sum(combi, [])
