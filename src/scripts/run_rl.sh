#!/bin/bash

NUM_NODES="$1"
NUM_ENV_RUNNERS="$2"

if [ -z "$NUM_NODES" ] || [ -z "$NUM_ENV_RUNNERS" ]; then
  echo "Usage: ./run_rl.sh <num_nodes> <num_env_runners>"
  exit 1
fi

# Install dependencies on all nodes
for node in $(cat "$OAR_NODEFILE" | sort | uniq); do
  echo "Installing dependencies on $node..."
  ssh "$node" "bash ~/rllib_lab/install_dependencies.sh"
done

# Pick the current node as head
HEAD_NODE=$(hostname)

# Start Ray head node
echo "Starting Ray head node on $HEAD_NODE..."
source ~/rllib_lab/rllib_venv/bin/activate 
ray start --head --port=6379 &

# Start Ray workers
for node in $(cat "$OAR_NODEFILE" | sort | uniq); do
  if [ "$node" != "$HEAD_NODE" ]; then
    echo "Starting Ray worker on $node..."
    ssh "$node" "source ~/rllib_lab/rllib_venv/bin/activate && ray start --address='$HEAD_NODE:6379'"
  fi
done

# Run Python script on head node
echo "Running Python script on head node..."
export NUM_NODES=$NUM_NODES
export NUM_ENV_RUNNERS=$NUM_ENV_RUNNERS
python3 ~/rllib_lab/rl.py

# Stop Ray on all nodes
for node in $(cat "$OAR_NODEFILE" | sort | uniq); do
  echo "Stopping Ray on $node..."
  ssh "$node" "ray stop"
done
