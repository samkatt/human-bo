import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from human_bo import factories


def save_results(path: str, **kwargs):
    """Saves the result in `path`

    :path: File path to store results
    :kwargs: any key -> val you would like to store
    :returns: NONE
    """
    torch.save(kwargs, path)


def eval_model(combi, budget, savefolder):
    """Main loop that handles all the work

    :combi:

    """
    exp, ker, acqf, n_init, seed = combi
    path = f"{savefolder}/{exp}_{ker}_{acqf}_{seed}.pt"
    torch.manual_seed(seed)

    problem = factories.pick_test_function(exp)
    bounds = torch.tensor(problem._bounds).T
    dim = bounds.shape[1]
    # TODO hardcoded noise level, but should be function-dependent.
    sigma = 0.01

    train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, dim)
    train_Y = problem(train_X).view(-1, 1)
    train_Y = train_Y + sigma * torch.randn(size=train_Y.shape)

    K = factories.pick_kernel(ker, dim)
    gpr = SingleTaskGP(train_X, train_Y, covar_module=K)
    mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
    fit_gpytorch_model(mll, max_retries=10)

    ##### BO LOOP
    regrets = torch.zeros(budget + 1)
    regrets[0] = problem.optimal_value - train_Y.max()
    for b in range(budget):
        af = factories.pick_acqf(acqf, train_Y, gpr, bounds)
        candidates, _ = optimize_acqf(
            acq_function=af,
            bounds=bounds,
            q=1,  # batch size, i.e. we only query one point
            num_restarts=10,
            raw_samples=512,
        )
        y = problem(candidates)
        train_X = torch.cat((train_X, candidates))
        train_Y = torch.cat((train_Y, y.view(-1, 1)))
        regrets[b + 1] = problem.optimal_value - train_Y.max()
        gpr = SingleTaskGP(train_X, train_Y, covar_module=K)
        mll = ExactMarginalLogLikelihood(gpr.likelihood, gpr)
        fit_gpytorch_model(mll, max_retries=10)

    save_results(path, train_X=train_X, train_Y=train_Y, regrets=regrets)


def parallel_eval(combi, budget, savefolder, x):
    """Runs `eval_model` for run `x`"""
    return eval_model(combi[x], budget, savefolder)
