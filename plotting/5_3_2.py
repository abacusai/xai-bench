import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# plt.rc('pgf', texsystem='pdflatex')
plt.style.use('science')
plt.rcParams["figure.figsize"] = (17,17)
import os
import pandas as pd
from collections import defaultdict
import numpy as np

mode="regression"
# datasets={"regression": ["gaussianLinear", "gaussianNonLinearAdditive", "gaussianPiecewiseConstant"]}
dataset = "gaussianPiecewiseConstant"
# metrics=["roar_faithfulness", "roar_monotonicity", "faithfulness", "monotonicity", "shapley"]
metrics=["roar_faithfulness", "roar_monotonicity", "faithfulness", "monotonicity", "shapley", "shapley_corr"]
metric_mappings = {
    "roar_faithfulness": "roar-faith-", 
    "roar_monotonicity": "roar-mono-", 
    "faithfulness": "faith-", 
    "monotonicity": "mono-",
    "shapley": "shapley-mse",
    "shapley_corr": "shapley-corr"
}
results_dir="../results/"
output_dir=f"plots/{mode}/error_bands/5_3_2/"
exp_names=["wed/exp-gaussian-new-1", "wed/exp-gaussian-new-2", "wed/exp-gaussian-new-3"]
# exp_names=["exp-mixture-1", "exp-mixture-2", "exp-mixture-3"]

# Important initializations
rhos = [0.0, 0.25, 0.5, 0.75, 0.99]
explainers = ["random", "shap", "shapr", "brutekernelshap", "maple", "lime", "l2x"]
explainer_mappings = {
    "random": "RANDOM", 
    "shap": "SHAP",
    "shapr": "SHAPR", 
    "brutekernelshap": "BF-SHAP",
    "maple": "MAPLE", 
    "lime": "LIME", 
    "l2x": "L2X"
}
models = ['LR', 'DTREE', 'MLP']
model_mappings = {
    'LR': 'linear regression', 
    'DTREE': 'decision tree', 
    'MLP': 'multilayer perceptron'
}

def collect_data(file_path, rho):
    lines = open(file_path, 'r').readlines()
    for idx, line in enumerate(lines):
        if line.strip().split(",")[0] in models:
            model = line.strip().split(",")[0]
            model_perfs[model][rho] = line.strip().split(",")[1:]
            df = pd.read_csv(file_path, skiprows=idx+1, nrows=len(explainers))
            df.columns.values[0] = "explainer"
            scores[model][rho].append(df)

for metric in metrics:
    num_models = len(models)
    fig, axs = plt.subplots(figsize=(6.0*num_models, 5.0), nrows=1, ncols=num_models)

    scores = {model: defaultdict(list) for model in models}
    model_perfs = {model: defaultdict(list) for model in models}

    for rho in rhos:
        for exp_name in exp_names:
            file_path = os.path.join(results_dir, mode, exp_name, "csv", f"{dataset}_{rho}.csv")
            collect_data(file_path, rho)

    for idx, model in enumerate(models):
        for explainer in explainers:
            values = [[] for _ in range(len(exp_names))] 
            for exp in range(len(exp_names)):
                values[exp] = [scores[model][r][exp].query(f'explainer=="{explainer}"')[metric] for r in rhos]
            values = np.squeeze(np.array(values))
            means = np.mean(values, axis=0)
            mins = np.min(values, axis=0)
            maxs = np.max(values, axis=0)
            axs[idx].plot(rhos, means, linewidth=2,  label = explainer)
            axs[idx].fill_between(rhos, mins, maxs, alpha=0.2)
        # axs2 = axs[idx].twinx()
        # axs[idx].set_yscale('log')
        axs[idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # axs2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        perfs = [float(model_perfs[model][r][1]) for r in rhos]
        # axs2.plot(rhos, perfs, linestyle='dashed', color='lightgrey', label='modelPerf')
        axs[idx].set_xlabel("Rho", fontsize=18)
        # axs[idx].set_ylabel(f"{metric} metric value")
        if idx == 0:
            axs[idx].set_ylabel(metric_mappings[metric], fontsize=18)
        axs[idx].set_title(model_mappings[model], fontsize=18)
        axs[idx].tick_params(axis='both', which='major', labelsize=18)
        if model == "DTREE":
            lines, labels = axs[idx].get_legend_handles_labels()
            lines2, labels2 = [] ,[]# axs2.get_legend_handles_labels()
            # axs[idx].legend(lines + lines2, labels + labels2, loc=0, fontsize=12)
                # axs[data_idx][idx].legend()  # bbox_to_anchor=(1.1, 1.05), loc="upper right")
    # fig.suptitle(f"Performance of explainers on the {metric} metric", fontsize=20)
    fig.subplots_adjust(bottom=0.22)
    fig.legend(lines + lines2, [explainer_mappings[a] for a in labels + labels2], frameon=True, shadow=True, loc="lower center", bbox_to_anchor=(0.43, 0), ncol=len(explainers), fontsize=14)
    plt.show()
    plt_save_path = os.path.join(output_dir, f"{dataset}_{metric}_all.pdf")
    if not os.path.exists(os.path.dirname(plt_save_path)):
        os.makedirs(os.path.dirname(plt_save_path))
    # fig.tight_layout()
    plt.savefig(plt_save_path, bbox_inches='tight')
    plt.clf()
    plt.cla()