"""Encodes different types of interactions as experiments."""

from typing import Any, Protocol


class Agent(Protocol):
    def pick_query(self) -> tuple[Any, dict[str, Any]]: ...
    def observe(self, query, feedback, evaluation) -> None: ...


class Problem(Protocol):
    def give_feedback(self, query) -> tuple[Any, dict[str, Any]]: ...
    def observe(self, query, feedback, evaluation) -> None: ...


class Evaluation(Protocol):
    def __call__(
        self,
        query,
        feedback,
        query_stats: dict[str, Any],
        feedback_stats: dict[str, Any],
        **kwargs
    ) -> tuple[Any, dict[str, Any]]: ...


class User(Protocol):
    def pick_action(self, query) -> tuple[Any, dict[str, Any]]: ...

    def observe(self, action, feedback, evaluation) -> None: ...


def basic_loop(
    agent: Agent, problem: Problem, evaluation_function: Evaluation, budget: int
):
    """Runs typical interleaving scheme where `agent` and `problem` are prompted one after the other"""

    evaluation_keys = [
        "evaluation",
        "evaluation_stats",
        "feedback",
        "feedback_stats",
        "query",
        "query_stats",
    ]

    evaluations: dict[str, Any] = {k: [] for k in evaluation_keys}

    for _ in range(budget):

        query, query_stats = agent.pick_query()
        feedback, feedback_stats = problem.give_feedback(query)

        evaluation, evaluation_stats = evaluation_function(
            query, feedback, query_stats, feedback_stats
        )

        agent.observe(query, feedback, evaluation)
        problem.observe(query, feedback, evaluation)

        for key, val in zip(
            evaluation_keys,
            [
                evaluation,
                evaluation_stats,
                feedback,
                feedback_stats,
                query,
                query_stats,
            ],
        ):
            evaluations[key].append(val)

    return evaluations
