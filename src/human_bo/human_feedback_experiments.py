"""Specific functions for human giving feedback experiments"""

from human_bo import conf


def update_config():
    """This function updates the configurations to set up for human feedback experiments

    This needs to be run at the start of any script on these type of experiments.
    """
    # Add `user_model` as a configuration.
    conf.CONFIG["user_model"] = {
        "type": str,
        "shorthand": "u",
        "help": "The mechanism through which queries are given.",
        "tags": {"experiment-parameter"},
        "parser-arguments": {"choices": {"oracle", "gauss"}},
    }
