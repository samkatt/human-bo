# Bayes Optimization with Human-in-the-Loop

In this setting we are interested in optimization for human preferences.
The core idea is to exploit multi-objective setting, without assuming the designer of the system knows all objectives a-priori.

## Installation

Install the package for typical usage:

```sh
python -m pip install .
```

## Running

Entry points are in `scripts`.
In particular, to run a simple Bayesian optimization problem, look at

```shell
for seed in $(seq 1 5); do python scripts/run_bo.py -s ${seed} -ni 3 -b 10 -a MES -p Zhou -e 0.1 -f results-dir --wandb scripts/wandb_example_config.yaml; done
```


Visualize the results with `scripts/visualize_human_ai_experiment.py`:

```sh
python scripts/visualize_human_ai_experiment.py results-dir/MES_Zhou_1.pkl 
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

Install and run `pytest`:

```sh
python -m pip install pytest  # or python -m pip install .'[test]'
python -m pytest
```

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

- [ ] Debug (visualizing) map. Seems to be stuck (e.g., in results on `python scripts/run_moo.py -t utility-learner -p BraninCurrin -a EI -f results -z 1`).
- [ ] Fix all tensorflow warnings.
- [ ] Remove references to `cost`.
- [ ] Check for consistency with:
    - [ ] Creation of tensorshapes.
    - [ ] Checking of tensorshapes.
    - [ ] Reshaping tensors.
