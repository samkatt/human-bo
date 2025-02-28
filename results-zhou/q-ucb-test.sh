#!/bin/bash
#SBATCH --time=00:10:00

#SBATCH --mem=100M

#SBATCH --cpus-per-task 1
#SBATCH --array=1-25

#SBATCH --output=out/%A-%a.out
#SBATCH --job-name=experiment-%A-%a

source ${WRKDIR}/init-environment.sh

set -u          # do not use empty strings when using unused variables.
set -o pipefail # return non-zero if any of the commands in a pipeline fails (not just last one).
set -x          # print commands before executing.

run_human_ai_experiment.py -s $SLURM_ARRAY_TASK_ID -b 20 -ni 0 -e 0.1 -p Zhou -f results-ucb -u oracle -a UCB --wandb q-ucb-test-wandb.yaml  $@
