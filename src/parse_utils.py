import argparse
import numpy as np
import random
import glob
import os
import re
import json
import logging
import pandas as pd
import dill as pickle

def valid_string(values):
    return f"Valid choices are: {list(values)}"


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def save_results(results: dict, results_dir: str):
    logFileName = f"{results['dataset']}"
    rho = str(results["dataset_kwargs"]["rho"]) if "rho" in results["dataset_kwargs"] else "na"
    saveName = f"{results_dir}/{logFileName}_{rho}.log"
    logging.info("Saving results in %s", saveName)
    if not os.path.exists(os.path.dirname(saveName)):
        os.makedirs(os.path.dirname(saveName))
    with open(saveName, "w") as f:
        f.write(json.dumps(results, indent=4))


def save_results_csv(results: dict, results_dir: str):
    logFileName = f"{results['dataset']}"
    rho = str(results["dataset_kwargs"]["rho"]) if "rho" in results["dataset_kwargs"] else "na"
    saveName = f"{results_dir}/csv/{logFileName}_{rho}.csv"
    logging.info("Saving results in %s", saveName)
    if not os.path.exists(os.path.dirname(saveName)):
        os.makedirs(os.path.dirname(saveName))
    with open(saveName, "w") as f:
        for model in results["models"]:
            f.write(str(model).upper() + "," + ",".join(results["model_perfs"][model]) + "\n")
            df = pd.read_json(json.dumps(results["models"][model]))
            df = df.transpose()
            f.write(df.to_csv())
            f.write("\n\n")

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def save_experiment(experiment, checkpoint_dir: str, rho):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    logFileName = f"{experiment.dataset.name}"
    saveName = f"{checkpoint_dir}/{logFileName}_{rho}.pkl"
    save_object(experiment, saveName)