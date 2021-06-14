<p align="center"><img src="img/banner.svg" width=700 /></p>

`XAI-Bench` is a library for benchmarking feature attribution techniques using synthetic data. Unlike real-world datasets, synthetic datasets allow the efficient computation of conditional expected values that are needed to evaluate many explainability metrics such as ground-truth Shapley values, faithfulness, and monotonicity. This repository can be used to benchmark six different feature attribution techniques across ten different evaluation metrics.

<p align="center"><img src="img/overview_figure.svg" width=700 /></p>


## Installation

To use our package, make sure you install all the dependencies in `requirements.txt` using 
```
pip install -r requirements.txt
```

## Usage (WIP)

The API can be accessed using `python main_driver.py`. 

```
> python main_driver.py --help
usage: Driver for the explainability project [-h] [--mode {classification,regression}] --dataset DATASET --model MODEL --explainer EXPLAINER [--metric METRIC]
                                             [--data-kwargs DATA_KWARGS | --data-kwargs-json DATA_KWARGS_JSON]
                                             [--model-kwargs MODEL_KWARGS | --model-kwargs-json MODEL_KWARGS_JSON] [--seed SEED] [--experiment] [--rho RHO]
                                             [--rhos RHOS [RHOS ...]] [--experiment-json EXPERIMENT_JSON] [--no-logs] [--results-dir RESULTS_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --mode {classification,regression}
                        Classification or regression?
  --dataset DATASET     Name of the dataset to train on
  --model MODEL         Algorithm to use for training
  --explainer EXPLAINER
                        Explainer to use
  --metric METRIC       Metric to evaluate the explanation
  --data-kwargs DATA_KWARGS
                        Custom data args needed to generate the dataset.\n Default = '{}'
  --data-kwargs-json DATA_KWARGS_JSON
                        Path to json file containing custom data args.
  --model-kwargs MODEL_KWARGS
                        Custom data args needed to generate the dataset.\n Default = '{}'
  --model-kwargs-json MODEL_KWARGS_JSON
                        Path to json file containing custom data args.
  --seed SEED           Setting a seed to make everything deterministic.
  --experiment          Run multiple experiments using an experiment config file.
  --rho RHO             Control the rho of an experiment.
  --rhos RHOS [RHOS ...]
                        Control the rhos of a mixture experiment.
  --experiment-json EXPERIMENT_JSON
  --no-logs             whether to save results or not. You can use this avoid overriding your result files while testing.
  --results-dir RESULTS_DIR
                        Path to save results in csv files.
```

### Sample Usage (WIP)

We use an `Experiment` to benchmark various datasets, models, explainers, metrics. This is the recommended way to access our library.

For explaining using an experiment config (recommended) on the regression datasets (recommended to use experiment config), use `--no-logs` when debugging
```
python main_driver.py --mode regression --seed 7 --experiment --experiment-json configs/experiment_config.json --no-logs
```
For running several experiments at once use a script as shown in,
```
./script.sh
```
Each `Experiment` object is saved after execution for checkpointing. This way, additional experiments can be run without having to rerun previous computation.

### Plotting (WIP)

Some scripts for visulizing the results are available in `plotting/`. You can use them to generate plots by the `results_dir` to your `results_dir`.
