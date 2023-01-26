# pylint: disable=R0801
import argparse

from datetime import datetime

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import pathlib

import sys
try:
    # try importing as if xai_bench is a module
    from .src import datasets, explainer, metric, experiments, model
except ImportError as e:
    if "relative import" in str(e):
        # try importing as if called from this file
        sys.path.append('../')
        from src import datasets, explainer, metric, experiments, model
    else:
        raise e

current_dir = pathlib.Path(__file__).parent
plt.style.use('https://raw.githubusercontent.com/RobGeada/stylelibs/main/material_rh.mplstyle')

"""
Define test configs
  - rhos: rho determines feature independence of generated datasets, part of metric grid
  - bench_datasets: datasets to bench, part of metric grid
  - bench_explainers: the explainers to benchmark per grid entry
    - each explainer has a name and set of kwargs to pass to explainer.__init__
  - bench_models: models to bench, part of metric grid
  - bench_metrics: metrics to include in grid
  - num_features: features per generated dataset
  - benchmark runs in a grid of rhos x datasets x explainers x models x metrics
"""
LEVEL_0_CONFIG = {
    "rhos": [.5],
    "bench_datasets": ["gaussianPiecewiseConstant"],
    "bench_explainers": {
        "shap_trustyai": {},
        "lime_trustyai": {"samples": 100, "normalise_weights": False, "separable_dataset_ratio": .9},
    },
    "bench_models": ["mlp"],
    "bench_metrics": ["roar_faithfulness", "roar_monotonicity", "faithfulness", "monotonicity",
                      "shapley", "shapley_corr", "infidelity"],
    "num_features": 3
}

LEVEL_1_CONFIG = {
    "rhos": np.linspace(0, 1, 5),
    "bench_datasets": ["gaussianLinear", "gaussianNonLinearAdditive", "gaussianPiecewiseConstant"],
    "bench_explainers": {
        "shap_trustyai": {},
        "shap": {},
        "lime_trustyai": {"samples": 100, "normalise_weights": False, "separable_dataset_ratio": .9},
        "lime": {"samples": 100},
        "random": {}
    },
    "bench_models": ["lr", "dtree", "mlp"],
    "bench_metrics": ["roar_faithfulness", "roar_monotonicity", "faithfulness", "monotonicity",
                      "shapley", "shapley_corr", "infidelity"],
    "num_features": 5
}

LEVEL_LIME_CONFIG = {
    "rhos": np.linspace(0, 1, 5),
    "bench_datasets": ["gaussianLinear", "gaussianNonLinearAdditive", "gaussianPiecewiseConstant"],
    "bench_explainers": {
        "lime_trustyai": {"samples": 100, "normalise_weights": False, "separable_dataset_ratio": .9},
        "lime": {"samples": 100},
        "random": {}
    },
    "bench_models": ["lr", "dtree", "mlp"],
    "bench_metrics": ["roar_faithfulness", "roar_monotonicity", "faithfulness", "monotonicity",
                      "shapley", "shapley_corr", "infidelity"],
    "num_features": 5
}

LEVEL_SHAP_CONFIG = {
    "rhos": np.linspace(0, 1, 5),
    "bench_datasets": ["gaussianLinear", "gaussianNonLinearAdditive", "gaussianPiecewiseConstant"],
    "bench_explainers": {
        "shap_trustyai": {},
        "shap": {},
        "random": {}
    },
    "bench_models": ["lr", "dtree", "mlp"],
    "bench_metrics": ["roar_faithfulness", "roar_monotonicity", "faithfulness", "monotonicity",
                      "shapley", "shapley_corr", "infidelity"],
    "num_features": 5
}

LEVEL_2_CONFIG = {
    "rhos": np.linspace(0, 1, 5),
    "bench_datasets": ["gaussianLinear", "gaussianNonLinearAdditive", "gaussianPiecewiseConstant",
                       "mixtureLinear", "mixtureNonLinearAdditive", "mixturePiecewiseConstant"],
    "bench_explainers": {
        "shap_trustyai": {},
        "shap": {},
        "lime_trustyai": {"samples": 100, "normalise_weights": False, "separable_dataset_ratio": .9},
        "lime": {"samples": 100, },
        "random": {}
    },
    "bench_models": ["lr", "dtree", "mlp"],
    "bench_metrics": ["roar_faithfulness", "roar_monotonicity", "faithfulness", "monotonicity",
                      "shapley", "shapley_corr", "infidelity"],
    "num_features": 7
}



configs = [LEVEL_0_CONFIG, LEVEL_1_CONFIG, LEVEL_2_CONFIG]
config_dict = {
    "0": LEVEL_0_CONFIG,
    "1": LEVEL_1_CONFIG,
    "2": LEVEL_2_CONFIG,
    "lime": LEVEL_LIME_CONFIG,
    "shap": LEVEL_SHAP_CONFIG
}

def run_test_config(test_config):
    """Run a particular test config"""
    data = []
    for dataset in test_config['bench_datasets']:
        for rho in test_config['rhos']:
            if "gaussian" in dataset:
                exp_dataset = datasets.Data(
                    name=dataset,
                    mode="regression",
                    mu=np.zeros(test_config['num_features']),
                    rho=rho,
                    dim=test_config['num_features'],
                    noise=0.01,
                    weight=np.arange(test_config['num_features'] - 1, -1, -1),
                    num_train_samples=1000,
                    num_val_samples=100)
            else:
                exp_dataset = datasets.Data(
                    name=dataset,
                    mode="regression",
                    mus=[np.ones(test_config['num_features']), (-1 * np.ones(test_config['num_features']))],
                    rho=rho,
                    dim=test_config['num_features'],
                    noise=0.01,
                    weight=np.arange(test_config['num_features'] - 1, -1, -1),
                    num_train_samples=1000,
                    num_val_samples=100)

            exp_models = [model.Model(name=m, mode="regression") for m in test_config['bench_models']]
            exp_explainers = [explainer.Explainer(name=e, **k) for e, k in test_config['bench_explainers'].items()]
            exp_metrics = [metric.Metric(name=m, conditional="observational")
                           for m in test_config['bench_metrics']]
            experiment_results = experiments.Experiment(
                exp_dataset,
                exp_models,
                exp_explainers,
                exp_metrics
            ).get_results()
            data += experiment_results
    return pd.DataFrame(data)


def plot_test_results(df, suffix):
    """Show the results of a benchmark as violin plots"""
    # get data information from df
    metrics = [x for x in list(df) if "metric" in x or x == 'runtime']
    metric_down = ["metric_infidelity", "metric_runtime", "metric_shapley", "runtime"]
    metric_labels = {m: m + (" ↓" if m in metric_down else " ↑") for m in metrics}
    explainers = df['explainer'].unique()
    n_exp = len(explainers)

    # setup plots
    cmap = mpl.colormaps['viridis']
    fig = plt.figure(figsize=(16, 9))
    labels = []

    # iterate through metrics
    for metric_idx, metric_name in enumerate(metrics):
        # iterate over explainers
        for exp_idx, explainer_name in enumerate(explainers):
            # slice dataframe over explainer
            exp_df = df[df['explainer'] == explainer_name]

            # grab metric data for this explainer
            color = cmap(exp_idx / n_exp)
            plt.subplot(2, 4, metric_idx + 1)
            xs = []
            ys = []
            for idx, (_, row) in enumerate(exp_df.iterrows()):
                xs.append(exp_idx)
                ys.append(row[metric_name])

            # plot metric data of this explainer
            plt.scatter(xs + ((np.random.rand(len(xs)) - .5) / 5) + .5, ys, c=[color] * len(xs), s=1)
            violin_parts = plt.violinplot(ys, positions=[exp_idx + .5], showmeans=True)

            # color violins per-explainer
            for part_name in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                violin_parts[part_name].set_color(color)
            for v_idx, pc in enumerate(violin_parts['bodies']):
                pc.set_facecolor(color)
                pc.set_edgecolor(color)

            # create legend labels
            if metric_idx == 0:
                labels.append([mpatches.Patch(color=color), explainer_name])

        # format plot
        ax = plt.gca()
        locs = np.arange(n_exp)
        ax.xaxis.set_ticks(locs, minor=True)
        ax.yaxis.set_ticks(np.array(plt.yticks()[0]), minor=True)
        ax.xaxis.set(ticks=locs + .5, ticklabels=explainers)
        ax.grid(True, which='minor', axis='x')
        ax.grid(False, which='major', axis='x')
        plt.xticks(rotation=45, ha='right')
        plt.title(
            metric_labels[metric_name]
                .replace("metric_", "")
                .replace("_", " ")
                .title()
                .replace("Roar", "ROAR")
        )

    # title and save
    plt.suptitle("TrustyAI XAIBench L{} @ {}".format(
        suffix,
        datetime.strftime(datetime.now(), "%H:%M %Y-%m-%d")
    ), color='k', fontsize=16)
    plt.tight_layout()
    fig.legend(*zip(*labels), loc='lower center', bbox_to_anchor=(.5, -.05))
    plt.savefig(
        current_dir / "results" / "plots" / "xai_bench_{}.pdf".format(suffix),
        bbox_inches='tight'
    )
    plt.show()


def run_benchmark_config(config, save=False, plot=False):
    """Run a benchmark config, optionally plot and save the results to file"""
    results_df = run_test_config(configs[config])
    if plot:
        plot_test_results(results_df, config)
    if save:
        results_df.to_pickle(current_dir / "results" / "level_{}_results.pkl".format(config))
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark explainers with xai-bench.')
    parser.add_argument("-c", "--config", type=str, required=True,
                        help='The benchmark config to run. See the readme for the details of each')
    parser.add_argument("-l", "--label", type=str,
                        help="A shorthand name for this benchmark run to remember it")

    args = parser.parse_args()
    suffix = args.config + "_" + args.label if args.label else args.config

    """Run a benchmark config, then plot and save the results to file"""
    results_df = run_test_config(config_dict[args.config.lower()])
    plot_test_results(results_df, suffix)
    results_df.to_pickle(current_dir / "results" / "level_{}_results.pkl".format(suffix))