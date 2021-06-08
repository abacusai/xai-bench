import pandas as pd
import numpy as np
import logging
import json
import os

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from synthetic_datasets import GaussianLinearRegression, GaussianLinearBinary
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from src import datasets, explainer, experiments, metric, model, parse_utils
from sklearn.metrics import mean_squared_error

mode = "regression"
results_dir = f"results/{mode}/thurs/exp-model-wine-shapr-3/"

models = ["lr",
    "dtree",
    "mlp"
]

explainers = [ #"shap",
    "shapr",
    # "kernelshap",
    # "brutekernelshap",
    # "random",
    # "lime",
    # "maple",
    # "l2x",
]

metrics = ["faithfulness", "monotonicity", "shapley", "shapley_corr"]

df = pd.read_csv('data/winequality-white.csv', sep=';')

X = df.drop('quality', axis=1)
X = preprocessing.StandardScaler().fit(X).transform(X)
y = df['quality']

X, X_val, y, y_val = train_test_split(X, y, test_size=0.01, random_state=7)
knn = KNeighborsRegressor(n_neighbors=1)
#knn = KNeighborsClassifier()
knn.fit(X,y)
# mse = np.mean((knn.predict(X) - y)**2)
# print('MSE: ',mse)
mean = np.mean(X, axis=0)
cov = np.cov(X, rowvar=False)

data_generator = datasets.Data("gaussianLinear", mode=mode, mu=repr(mean), dim=len(mean), noise=0.01, 
                                            sigma=repr(cov), weight=repr(np.ones(len(mean))))


def make_experiment_with_dataset(dataset, models, explainers, metrics):
    models = [
        model.Model(mod, mode)
        for mod in models
    ]
    explainers = [
        explainer.Explainer(expl) for expl in explainers
    ]
    metrics = [metric.Metric(metr) for metr in metrics]
    return experiments.Experiment(dataset, models, explainers, metrics)


experiment = make_experiment_with_dataset(data_generator, models, explainers, metrics)
results = experiment.get_results()
logging.info(f"\nExperiment results : {json.dumps(results, indent=4)}")

parse_utils.save_experiment(experiment, os.path.join(results_dir, "checkpoints"), "na")
parse_utils.save_results(results, results_dir)
parse_utils.save_results_csv(results, results_dir)


# synthetic_samples, _ = data_generator.get_dataset(num_samples=len(X))
# y_synthetic = knn.predict(synthetic_samples)

# X, X_val = pd.DataFrame(X), pd.DataFrame(X_val)

# model_real = MLPRegressor().fit(X, y)
# model_syn = MLPRegressor().fit(synthetic_samples, y_synthetic)

# def get_real_syn_explanations_mse(real, syn):
#     return mean_squared_error(real, syn)

# for explainer_name in explainers:
#     explainer_real = explainer.Explainer(explainer_name).explainer(model_real.predict, X)
#     explainer_syn = explainer.Explainer(explainer_name).explainer(model_syn.predict, synthetic_samples)
#     feature_weights_real = explainer_real.explain(X_val)
#     feature_weights_syn = explainer_syn.explain(X_val)
#     mse = get_real_syn_explanations_mse(feature_weights_real, feature_weights_syn)
#     print(f"{explainer_name} mse is: {mse}")


# shap mse is: 0.0396764547060464
# brutekernelshap mse is: 0.05015404817057441
# random mse is: 1.8889475270676486
# lime mse is: 0.03713335058317393
# maple mse is: 0.06605651721865091
# l2x mse is: 0.0008952742128657123