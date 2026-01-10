#!/bin/bash

NUM_NODES="$1"
NUM_ENV_RUNNERS="$2"
WALLTIME="02:00:00"

if [ -z "$NUM_NODES" ] || [ -z "$NUM_ENV_RUNNERS" ]; then
  echo "Usage: ./run_job.sh <num_nodes> <num_env_runners>"
  exit 1
fi

oarsub -l host=$NUM_NODES,walltime=$WALLTIME "./run_rl.sh $NUM_NODES $NUM_ENV_RUNNERS"