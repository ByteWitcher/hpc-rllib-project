import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
import json

NUM_NODES = int(os.environ.get("NUM_NODES", 1))
NUM_ENV_RUNNERS = int(os.environ.get("NUM_ENV_RUNNERS", 1))

ray.init(address="auto") 

config = (
    PPOConfig()
    .environment("Taxi-v3")
    .env_runners(
        num_env_runners=NUM_ENV_RUNNERS,
        env_to_module_connector=lambda env: FlattenObservations(),
    )
    .evaluation(evaluation_num_env_runners=1)
)

algo = config.build_algo()

log_file = "results.jsonl"

for i in range(10):
    result = algo.train()

    # Extract timers
    timers = result.get("timers", {})
    sample_time_s = timers.get("env_runner_sampling_timer", 0)
    learn_time_s = timers.get("learner_update_timer", 0)
    total_time_s = result.get("time_this_iter_s", 0)

    # Compute throughput (samples per second)
    num_steps = result.get("num_training_step_calls_per_iteration", 0)
    throughput_sps = num_steps / sample_time_s if sample_time_s > 0 else 0

    log = {
        "nodes": NUM_NODES,
        "num_env_runners": NUM_ENV_RUNNERS,
        "iteration": result.get("training_iteration"),
        "total_time_s": total_time_s,
        "sample_time_s": sample_time_s,
        "learn_time_s": learn_time_s,
        "throughput_sps": throughput_sps,
        "cpu_util_percent": result.get("perf", {}).get("cpu_util_percent", None),
        "ram_util_percent": result.get("perf", {}).get("ram_util_percent", None)
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(log) + "\n")

algo.stop()