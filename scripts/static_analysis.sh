#!/usr/bin/env bash
set -u          # do not use empty strings when using unused variables.
set -o pipefail # return non-zero if any of the commands in a pipeline fails (not just last one).
set -x          # print commands before executing.

black .
flake8 human_bo tests
mypy .
prospector
python -m pytest .
