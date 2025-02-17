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


def basic_interleaving(
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


def ai_advices_human_loop(
    agent: Agent,
    user: User,
    problem: Problem,
    evaluation_function: Evaluation,
    budget: int,
) -> dict[str, list]:
    """Main loop for AI suggestion then Human pick joint optimization

    Pretty straightforward interactive experiment setup:
        1. Ask action from `ai`
        2. Ask action from `human` _given AI action_
        3. Apply both actions to `problem`

    The actual implementation depends heavily on how `ai`, `human`, and `problem` are implemented!
    """

    evaluation_keys = [
        "evaluation",
        "evaluation_stats",
        "feedback",
        "feedback_stats",
        "ai_query",
        "ai_stats",
        "query",
        "query_stats",
    ]

    evaluations: dict[str, Any] = {k: [] for k in evaluation_keys}

    k = 0
    while k < budget:

        ai_query, ai_stats = agent.pick_query()
        user_query, user_stats = user.pick_action(ai_query)
        feedback, feedback_stats = problem.give_feedback(user_query)

        evaluation, evaluation_stats = evaluation_function(
            user_query,
            feedback,
            user_stats,
            feedback_stats,
            ai_query=ai_query,
            ai_stats=ai_stats,
        )

        agent.observe(
            ai_query, {"action": user_query, "feedback": feedback}, evaluation
        )
        user.observe(
            user_query, {"ai_query": ai_query, "feedback": feedback}, evaluation
        )
        problem.observe(
            user_query, {"ai_query": ai_query, "feedback": feedback}, evaluation
        )

        for key, val in zip(
            evaluation_keys,
            [
                evaluation,
                evaluation_stats,
                feedback,
                feedback_stats,
                ai_query,
                ai_stats,
                user_query,
                user_stats,
            ],
        ):
            evaluations[key].append(val)

        k = k + user_query.shape[0]

    return evaluations
