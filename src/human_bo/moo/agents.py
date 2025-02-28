"""These are implementations of agents in MOO.

They mostly differ in the type of prior knowledge and inference they do on the
data.

"""

from typing import Any

import torch
from botorch import fit, models, optim
from botorch.acquisition import objective
from botorch.models.transforms import input as input_transform
from botorch.models.transforms import outcome as outcome_transform
from gpytorch.mlls import sum_marginal_log_likelihood

from human_bo import core, interaction_loops
from human_bo.moo import moo_core


class UtilityBO(interaction_loops.Agent):
    def __init__(self, bounds, kernel: str, acqf: str, acqf_options: dict[str, Any]):
        self.bo = core.PlainBO(kernel, acqf, bounds, acqf_options)

        self.x = torch.Tensor()
        self.u = torch.Tensor()

    def pick_query(self) -> tuple[Any, dict[str, Any]]:
        query, acqf_val = self.bo.pick_queries(self.x, self.u)
        return query, {"acqf_value": acqf_val}

    def observe(self, query, feedback, evaluation) -> None:
        del evaluation

        self.x = torch.cat((self.x, query))
        self.u = torch.cat((self.u, feedback["utility"]))


class ObjectiveLearner(interaction_loops.Agent):
    """An agent that knows the utility function but must learn all objectives.

    Knows the utility function is a linear combination of the multiple
    objectives (and their weights). Observes all objectives and utilities, and
    aims to pick queries that optimize acquisition function over utility
    directly.

    Maintains a list of GPs as surrogates for the objectives.
    """

    def __init__(
        self,
        utility: moo_core.UtilityFunction,
        bounds,
        kernel: str,
        acqf: str,
        num_objs: int,
        acqf_options: dict[str, Any],
    ):
        assert num_objs > 1

        self.kernel = kernel
        self.acqf = acqf
        self.acqf_options = acqf_options

        self.bounds = bounds
        self.dim = bounds.shape[1]
        self.num_objs = num_objs

        self.utility_function = utility

        self.x = torch.Tensor()
        self.o = torch.Tensor()
        self.u = torch.Tensor()

    def pick_query(self) -> tuple[Any, dict[str, Any]]:

        # Base case: there is no data -> return random point.
        if self.x.nelement() == 0:
            print(
                "WARN: PlainBO::pick_queries is returning randomly because of empty x."
            )
            return core.random_queries(self.bounds), {"acqf_value": torch.Tensor(0)}

        # Fit models to data.
        gprs = [
            models.SingleTaskGP(
                self.x,
                self.o[:, i].unsqueeze(-1),
                covar_module=core.pick_kernel(self.kernel, self.dim),
                input_transform=input_transform.Normalize(
                    d=self.dim, bounds=self.bounds
                ),
                outcome_transform=outcome_transform.Standardize(m=1),
            )
            for i in range(self.num_objs)
        ]
        model = models.ModelListGP(*gprs)
        fit.fit_gpytorch_mll(
            mll=sum_marginal_log_likelihood.SumMarginalLogLikelihood(
                model.likelihood, model
            )
        )

        # Run acquisition function on models.
        acqf_func = moo_core.create_acqf(
            self.acqf,
            model,
            objective.GenericMCObjective(lambda Y, X: self.utility_function(Y)),
            self.x,
            **self.acqf_options,
        )

        candidates, acqf_val = optim.optimize_acqf(
            acqf_func,
            bounds=self.bounds,
            q=1,  # batch size, i.e. we only query one point
            num_restarts=10,
            raw_samples=512,
        )
        return candidates, {"acqf_value": acqf_val}

    def observe(self, query, feedback, evaluation) -> None:
        del evaluation

        self.x = torch.cat((self.x, query))
        self.u = torch.cat((self.u, feedback["utility"]))

        self.o = torch.cat(
            (
                self.o,
                torch.tensor(
                    [list(q.values()) for q in feedback["objectives"].values()]
                ),
            )
        )


def create_AI(
    moo_function,
    utility_function: moo_core.UtilityFunction,
    algorithm: str,
    kernel: str,
    acqf: str,
    acqf_options: dict[str, Any],
) -> interaction_loops.Agent:
    """Creates an AI agent for MOO.

    This is where main difference in experiments happen, e.g.:
        - algorithm == "random" returns a random query agent.
        - algorithm == "bo" returns a BO (on utility) agent.
        - algorithm == "objective-learner" learns objective but knows utility.
    """
    if algorithm == "random":
        return core.RandomAgent(moo_function.bounds)
    if algorithm == "bo":
        return UtilityBO(moo_function._bounds, kernel, acqf, acqf_options)
    if algorithm == "objective-learner":
        return ObjectiveLearner(
            utility_function,
            moo_function.bounds,
            kernel,
            acqf,
            moo_function.num_objectives,
            acqf_options,
        )

    raise ValueError(f"Algorithm {algorithm} is not supported.")
