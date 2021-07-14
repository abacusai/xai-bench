import sys
import os
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

import argparse
import commentjson
import numpy as np
import json
from src import datasets, model, explainer, metric, experiments, parse_utils


def get_args():
    parser = argparse.ArgumentParser("Driver for the explainability project")
    parser.add_argument(
        "--mode",
        default="regression",
        choices=["classification", "regression"],
        help="Classification or regression?",
    )
    parser.add_argument(
        "--dataset",
        required="--experiment" not in sys.argv,
        help="Name of the dataset to train on",
    )
    parser.add_argument(
        "--model",
        required="--experiment" not in sys.argv,
        help="Algorithm to use for training",
    )
    parser.add_argument(
        "--explainer", required="--experiment" not in sys.argv, help="Explainer to use"
    )
    parser.add_argument(
        "--metric", default="faithfulness", help="Metric to evaluate the explanation"
    )
    data_kwargs_group = parser.add_mutually_exclusive_group()
    data_kwargs_group.add_argument(
        "--data-kwargs",
        default={},
        type=commentjson.loads,
        help=r"Custom data args needed to generate the dataset.\n Default = '{}' ",
    )
    data_kwargs_group.add_argument(
        "--data-kwargs-json",
        default={},
        type=str,
        help=r"Path to json file containing custom data args.",
    )
    model_kwargs_group = parser.add_mutually_exclusive_group()
    model_kwargs_group.add_argument(
        "--model-kwargs",
        default={},
        type=commentjson.loads,
        help=r"Custom data args needed to generate the dataset.\n Default = '{}' ",
    )
    model_kwargs_group.add_argument(
        "--model-kwargs-json",
        default={},
        type=str,
        help=r"Path to json file containing custom data args.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Setting a seed to make everything deterministic.",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run multiple experiments using an experiment config file.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        help="Control the rho of an experiment.",
    )
    parser.add_argument(
        "--rhos",
        nargs="+",
        help="Control the rhos of a mixture experiment.",
    )
    parser.add_argument(
        "--experiment-json", required="--experiment" in sys.argv, type=str, help=""
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="whether to save results or not. You can use this avoid overriding your result files while testing.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/logs/",
        type=str,
        help="Path to save results in csv files.",
    )
    args = parser.parse_args()
    if args.data_kwargs_json:
        args.data_kwargs = commentjson.load(open(args.data_kwargs_json))
    if args.seed:
        parse_utils.set_global_seed(args.seed)
    if args.experiment_json:
        args.experiment_json = commentjson.load(open(args.experiment_json))
    return args


def process_args(args):
    if args.experiment_json:
        if args.rho is not None:
            args.experiment_json["dataset"]["data_kwargs"]["rho"] = args.rho
        if args.rhos is not None:
            args.experiment_json["dataset"]["data_kwargs"]["rhos"] = [float(rho) for rho in args.rhos]
        if args.dataset is not None:
            args.experiment_json["dataset"]["name"] = args.dataset
        metric_kwargs = {}
        if "conditional" in args.experiment_json:
            metric_kwargs['conditional'] = args.experiment_json['conditional']

        logging.info(
            f'\n Dataset config is: {json.dumps(args.experiment_json["dataset"])}'
        )
        dataset = datasets.Data(
            args.experiment_json["dataset"]["name"],
            args.mode,
            **args.experiment_json["dataset"]["data_kwargs"],
        )
        models = [
            model.Model(mod["name"], args.mode, **mod["model_kwargs"])
            for mod in args.experiment_json["models"]
        ]
        explainers = [
            explainer.Explainer(expl["name"], **expl["expl_kwargs"]) for expl in args.experiment_json["explainers"]
        ]
        metrics = [metric.Metric(metr, **metric_kwargs) for metr in args.experiment_json["metrics"]]
        return experiments.Experiment(dataset, models, explainers, metrics)

    dataset = datasets.Data(args.dataset, args.mode, **args.data_kwargs)
    models = model.Model(args.model, args.mode, **args.model_kwargs)
    explainers = explainer.Explainer(args.explainer)
    metrics = metric.Metric(args.metric)
    return experiments.Experiment(dataset, [models], [explainers], [metrics])


if __name__ == "__main__":
    args = get_args()
    experiment = process_args(args)
    results = experiment.get_results()
    logging.info(f"\nExperiment results : {json.dumps(results, indent=4)}")
    if not args.no_logs:
        parse_utils.save_experiment(experiment, os.path.join(args.results_dir, "checkpoints"), args.rho)
        parse_utils.save_results(results, args.results_dir)
        parse_utils.save_results_csv(results, args.results_dir)
