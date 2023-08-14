# Bayes optimization with human-in-the-loop

## Installation

Install the required dependencies:

```
python -m pip install -r requirements.txt
```

## Running

### From command line

Example execution:

```
mkdir result-dir
for i in $(seq 1 5); do python scripts/main.py -n 2 -ni 3 -b 20 -k RBF -a MES -f Zhou -se $i -s result-dir; end
for i in $(seq 1 5); do python scripts/main.py -n 2 -ni 3 -b 20 -k RBF -a UCB -f Zhou -se $i -s result-dir; end
```

Then visualize by giving the files. There are two visualizations supported right now.
Either compare regrets (aggregates mean over different seed):

```
python scripts/visualize.py -t regrets result-dir/*
```

Or generate the "end" result, mostly for debugging:

```
python scripts/visualize.py -t end-result result-dir/Zhou_RBF_MES_1.pt result-dir/Zhou_RBF_MES_2.pt
```

### Tests

Install and run `pytest`:

```
python -m pip install pytest
pytest
```

## Data

Each run will generate a single result file called
`<some-unique-identifiers>_seed>.pt`. This contains, among other things, the
configurations of the run. In order to visualize, provide all files you'd like
to include in the comparison and the script will figure out how to combine them
(aggregate over runs with the same prefix). It will also try to ensure that
configurations are equal (e.g. budget is the same).
