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
python scripts/main.py -n 1 -ni 6 -b 200 -k RBF -a UCB -e Zhou -se 10 -s result-dir

```

### Tests

Install and run `pytest`:

```
python -m pip install pytest
pytest
```
