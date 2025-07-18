import json
from pathlib import Path
import matplotlib.pyplot as plt

# from plotly.subplots import make_subplots
# from plotly.graph_objects import Scatter
import numpy as np
from argparse import ArgumentParser
import pandas as pd

plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 
                     'font.size': 10,
                     })
plt.style.use('seaborn-v0_8-whitegrid')

cur_dir = Path(__file__).parent.resolve()
figs_dir = cur_dir / "figures"
figs_dir.mkdir(exist_ok=True)
results_dir = cur_dir / "results"


assert (cur_dir / "concept_activations.json").is_file(), "Concept activations file not found."


model_names = {
    'qwen2.5-coder': 'Code LLM',
    'qwen2.5': 'Generic LLM',
    'qwq': 'Reasoning LLM',
}

parser = ArgumentParser(description="Plot RQ4 Results")
parser.add_argument("-m",'--models', type=str, nargs='+', default=['Qwen2.5-Coder', 'Qwen2.5', 'QwQ'], help='List of model names to analyze')
parser.add_argument('--mode', type=str, default='individual', choices=['all', 'individual'], help='Mode of analysis')

args = parser.parse_args()

x = np.arange(64)


fig, axs = plt.subplots(1, 4, sharey=True, figsize=(24, 3), constrained_layout=True)

model_dfs = []
for model in args.models:
    try:
        filepath = results_dir / f"{model}_concept_activations_all.json" if args.mode == 'all' else results_dir / f"{model}_concept_activations.json"
        with open(filepath, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"File not found for model: {model}. Skipping...")
        continue
    
    report = {}

    for i, comment_type in enumerate(all_data):
        row = i + 1
        report[comment_type] = {}

        for j, benchmark in enumerate(all_data[comment_type]):
            col = j + 1
            report[comment_type][benchmark] = {}

            for task in all_data[comment_type][benchmark]:
                y = all_data[comment_type][benchmark][task]
                if benchmark == 'refinement':
                    print(comment_type, len(y), len(y[0]) if y else 0)

                # if not y:
                #     continue
                # if not isinstance(y, list):
                #     y = [y]
                y = np.array(y).mean(axis=0) if y and isinstance(y, list) else np.zeros(64)
                report[comment_type][benchmark][task] = float(np.mean(y))
                
                # axs[i].plot(
                #         y,
                #         marker=',',
                #         linestyle=linestyle,
                #         label=f"Benchmark {benchmark} - {comment_type} - {task}: {np.max(y):.4f}",
                # )
                
                # axs[i].set_title(f"{model_names[model_name]}")


            benchmark_df = pd.DataFrame(report[comment_type][benchmark], index=["Maximum Concept Activation Across Layers"]).T
            benchmark_df.index.name = 'Task'
            benchmark_df.reset_index(inplace=True)
            benchmark_df.sort_values(by="Maximum Concept Activation Across Layers", inplace=True, ascending=False)
            benchmark_df.to_csv(results_dir / f"{model}_{benchmark}_report.csv", index=False)

    # fig.update_layout(
    #     title=f"Probability of Comment Types in {model} across Tasks",
    #     xaxis_title="Layer Index",
    #     yaxis_title="Activation Probability",
    #     height=2000,
    #     width=2000,
    #     legend_title="Tasks",
    #     showlegend=True,
    # )
    # fig.write_image(figs_dir / f"{model}_all_tasks.pdf")
    # try:
    #     fig.write_html(figs_dir / f"{model}_all_tasks.html")
    # except Exception as e:
    #     print(f"Error writing HTML for {model}: {e}")
    #     continue

    df = pd.DataFrame(report)
    df.to_csv(results_dir / f"{model}_report.csv")

    flat_records = []
    for comment_type, bdict in report.items():
        for benchmark, tdict in bdict.items():
            for task, value in tdict.items():
                flat_records.append({
                    'CommentType': comment_type,
                    'Task': task,
                    'Value': value,
                })

    df = pd.DataFrame(flat_records)
    pivot_df = df.pivot(index="Task", columns="CommentType", values="Value")
    pivot_df.to_csv(results_dir / f"{model}_report_flat.csv")

    model_dfs.append(pivot_df.round(2))


if len(model_dfs) >= 3:
    combined_df = model_dfs[0].astype(str) + " / " + model_dfs[1].astype(str) + " / " + model_dfs[2].astype(str)
elif len(model_dfs) == 2:
    combined_df = model_dfs[0].astype(str) + " / " + model_dfs[1].astype(str)
elif len(model_dfs) == 1:
    combined_df = model_dfs[0]
else:
    combined_df = pd.DataFrame()
combined_df.to_csv(results_dir / "combined_report.csv", index=True)