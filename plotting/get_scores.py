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
datasets={"regression": ["gaussianPiecewiseConstant"]}
metrics=["roar_faithfulness", "roar_monotonicity", "faithfulness", "monotonicity"]#, "shapley", "shapley_corr"]
# metrics=["faithfulness", "monotonicity", "shapley", "shapley_corr"]
results_dir="../results/"
output_dir=f"plots/{mode}/error_bands/"
exp_names=["wed/exp-gaussian-new-1", "wed/exp-gaussian-new-2", "wed/exp-gaussian-new-3"]
# exp_names=["exp-mixture-1", "exp-mixture-2", "exp-mixture-3"]

# Important initializations
rhos = [0.0]
explainers = ["random", "shap", "brutekernelshap", "shapr", "lime", "maple", "l2x"]
models = ['MLP']

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
    num_models, num_datasets = len(models), len(datasets[mode])
    fig, axs = plt.subplots(figsize=(5.0*num_models, 4.0*num_datasets), nrows=num_datasets, ncols=num_models)

    for data_idx, dataset in enumerate(datasets[mode]):
        scores = {model: defaultdict(list) for model in models}
        model_perfs = {model: defaultdict(list) for model in models}

        for rho in rhos:
            for exp_name in exp_names:
                file_path = os.path.join(results_dir, mode, exp_name, "more_csv/csv/", f"{dataset}_{rho}.csv")
                collect_data(file_path, rho)

        for idx, model in enumerate(models):
            s_out = ""
            for explainer in explainers:
                values = [[] for _ in range(len(exp_names))] 
                for exp in range(len(exp_names)):
                    values[exp] = [scores[model][r][exp].query(f'explainer=="{explainer}"')[metric] for r in rhos]
                values = np.squeeze(np.array(values))
                means = np.mean(values, axis=0)
                mins = np.min(values, axis=0)
                maxs = np.max(values, axis=0)
                stds = np.std(values, axis=0)
                print(round(means, 3), "\pm", round(stds, 3), explainer, metric)
                s_out += f"& ${round(means, 3)}{{\\pm{round(stds, 3)}}}$" + "    "
            print(s_out)
            import pdb; pdb.set_trace()