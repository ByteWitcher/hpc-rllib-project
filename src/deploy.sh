#!/bin/bash

SITE="$1"

if [ -z "$SITE" ]; then
  echo "Usage: ./deploy.sh <site>"
  exit 1
fi

if [ ! -f scripts/run_job.sh ] || [ ! -f scripts/run_rl.sh ] || [ ! -f scripts/install_dependencies.sh ] || [ ! -f rl/rl.py ]; then
  echo "One or more files are missing"
  exit 1
fi

# List of configurations: nodes x env_runners
CONFIGS=("1:1" "1:2" "1:3" "2:1" "2:2" "2:3" "3:1" "3:2" "3:3")

# Deploy files to grid5000
echo "Creating directory rllib_lab on $SITE..."
ssh $SITE.g5k "mkdir -p ~/rllib_lab"

echo "Copying files to $SITE..."

scp scripts/run_job.sh $SITE.g5k:~/rllib_lab/run_job.sh
scp scripts/run_rl.sh $SITE.g5k:~/rllib_lab/run_rl.sh
scp scripts/install_dependencies.sh $SITE.g5k:~/rllib_lab/install_dependencies.sh
scp rl/rl.py $SITE.g5k:~/rllib_lab/rl.py

echo "All files deployed"

# Run jobs
for cfg in "${CONFIGS[@]}"; do
    IFS=":" read -r NUM_NODES NUM_ENV_RUNNERS <<< "$cfg"
    echo "Submitting job for Nodes=$NUM_NODES, Env Runners=$NUM_ENV_RUNNERS ..."

    JOB_ID=$(ssh $SITE.g5k "cd ~/rllib_lab && bash run_job.sh $NUM_NODES $NUM_ENV_RUNNERS" | grep "OAR_JOB_ID" | cut -d= -f2)
    echo "Submitted job $JOB_ID"

    while true; do
        JOB_STATUS=$(ssh $SITE.g5k "oarstat -s -j $JOB_ID")
        if [[ "$JOB_STATUS" =~ ": Terminated" ]]; then
            echo "Job $JOB_ID finished."
            break
        fi
        sleep 5
    done
done

# Copying results from grid5000 to our pc
scp $SITE.g5k:~/rllib_lab/results.jsonl rl/results.jsonl

cd plot
source ../../venv/bin/activate
python3 plot.py