# Bayes optimization with human-in-the-loop

In this setting we are interested in optimization of some (initially) unknown function by collaboration between an AI and human user.
The main motivation is that the human may have domain knowledge and the AI can do principled Bayesian optimization, and that we should be able to combine both.
The core research question we try to answer is how (artificial) theory of mind can help in this case.

## Core Experiments

On a high level, this code base allows you to run experiments and inspect its results.
Since there are many choices to be made, e.g. what function to optimize, the configuration is quite extensive.
However, the fundamental experiment is to minimize `regret` by optimizing `AI` given some function `f` and human `Human` as follows:

```python
for i in range(budget):
    x_ai[i] = AI(...)
    x_h[i] = Human(..., x)
    y[i] = [f(x_ai), f(x_h)]
    regret[i] = y_max - max(flatten(y[i]))
```

## Installation

Install the package for typical usage:

```sh
python -m pip install .
```

## Running

Entry points are in `scripts`.
In particular, to run a simple Bayesian optimization problem, look at

```shell
for seed in $(seq 1 5); do python scripts/run_human_ai_experiment.py -s ${seed} -ni 3 -b 10 -k RBF -a MES -f Zhou -e 0.1 -u oracle -f results-dir; done
```

Then visualize by giving the files. There are two visualizations supported right now.
Either compare regrets (aggregates mean over different seed):

```sh
python scripts/visualize.py -t regrets -f result-dir/*
```

Or generate the full trajectory of a single run, in the case of 1-D optimization, mostly for debugging:

```sh
python scripts/visualize.py -t trajectory -f result-dir/Zhou_RBF_MES_1.pt
```

## Tests

Install and run `pytest`:

```sh
python -m pip install pytest  # or python -m pip install .'[test]'
python -m pytest
```

## Data

Each run will generate a single result file called `<some-unique-identifiers>_seed>.pt`.
This contains, among other things, the configurations of the run.
In order to visualize, provide all files you'd like to include in the comparison and the script will figure out how to combine them (aggregate over runs with the same prefix).
It will also try to ensure that configurations are equal (e.g. budget is the same).

## Wandb

There is first-citizen support for experiments with [Wandb](https://www.wandb.ai).
Wandb requires you to specify the `project` and `entity`, but other options as well (see [example config file](scripts/wandb_example_config.yaml)).
To enable wandb, simply provide the configuration file: `python scripts/pick-a-script.py --wandb scripts/wandb_example_config.yaml ...`.

## Development

I recommend to install some packages to help with development (see `pyproject.toml`):

```sh
python -m pip install .'[dev]'
```

Try to keep the formatting consistent with `black .`

Basic linting includes:

```sh
mypy .
flake8 scripts src tests
```

But I tend to just run `scripts/static_analysis.sh` and check the output.

### To do

- [ ] Figure out first experiment: random vs human BO.
- [ ] Fix (understanding of) negation in test functions.
- [ ] Add visualisation to human-then-AI experiment.
- [?] Update configuration
    - [ ] Accept configuration file.
    - [ ] Accept overwrites.
