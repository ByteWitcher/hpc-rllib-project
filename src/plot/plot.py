import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams.update({'figure.dpi': 120})

results_path = os.path.join("..", "rl", "results.jsonl")
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Load JSONL data
records = []
with open(results_path, "r") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)
print("Loaded data:")
print(df.head())

# Aggregate metrics per configuration (nodes x num_env_runners)
agg = df.groupby(['nodes', 'num_env_runners']).agg({
    'total_time_s': 'mean',
    'sample_time_s': 'mean',
    'learn_time_s': 'mean',
    'throughput_sps': 'mean',
    'cpu_util_percent': 'mean',
    'ram_util_percent': 'mean'
}).reset_index()
print("\nAggregated metrics per configuration:")
print(agg)

# Total Iteration Time vs Nodes (per env runner)
plt.figure(figsize=(8,5))
sns.lineplot(data=agg, x='nodes', y='total_time_s', hue='num_env_runners', marker='o')
plt.xlabel("Number of Nodes")
plt.ylabel("Average Total Iteration Time (s)")
plt.title("Scaling: Total Iteration Time vs Nodes")
plt.legend(title="Num Env Runners")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "total_time_vs_nodes.png"))

# Throughput vs Nodes (per env runner)
plt.figure(figsize=(8,5))
sns.lineplot(data=agg, x='nodes', y='throughput_sps', hue='num_env_runners', marker='o')
plt.xlabel("Number of Nodes")
plt.ylabel("Throughput (Samples/sec)")
plt.title("Scaling: Throughput vs Nodes")
plt.legend(title="Num Env Runners")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "throughput_vs_nodes.png"))

# Iteration Time Breakdown per nodes/runners
for (nodes, runners), group in df.groupby(['nodes', 'num_env_runners']):
    plt.figure(figsize=(8,5))
    plt.bar(group['iteration'] - 0.2, group['sample_time_s'], width=0.4, color='skyblue', label='Sample')
    plt.bar(group['iteration'] - 0.2, group['learn_time_s'], bottom=group['sample_time_s'], width=0.4, color='salmon', label='Learn')
    plt.xlabel("Iteration")
    plt.ylabel("Time per Iteration (s)")
    plt.title(f"Iteration Time Breakdown (Sample vs Learn)\nNodes: {nodes}, Env Runners: {runners}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"iteration_time_nodes{nodes}_runners{runners}.png"))
    plt.close()

# CPU / RAM Utilization per nodes/runners
for (nodes, runners), group in df.groupby(['nodes', 'num_env_runners']):
    plt.figure(figsize=(8,5))
    sns.lineplot(data=group, x='iteration', y='cpu_util_percent', label='CPU')
    sns.lineplot(data=group, x='iteration', y='ram_util_percent', label='RAM')
    plt.xlabel("Iteration")
    plt.ylabel("Resource Usage (%)")
    plt.title(f"CPU / RAM Utilization\nNodes: {nodes}, Env Runners: {runners}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cpu_ram_nodes{nodes}_runners{runners}.png"))
    plt.close()

print("\nAll plots saved in the 'plots/' folder, grouped per nodes/env runners.")
