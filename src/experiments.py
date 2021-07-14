import shap
import numpy as np
from custom_explainers import GroundTruthShap
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, mean_squared_error

class DatasetAsModel():
    def __init__(self, data_class) -> None:
        self.data_class = data_class
        self.name = "dataset"
    def predict(self, X):
        # import pdb; pdb.set_trace()
        return self.data_class.generatetarget(X)
    def train(self, X, y):
        return DatasetAsModel(self.data_class)

class Experiment:
    def __init__(
        self, dataset, models, explainers, metrics
    ):
        self.dataset = dataset
        self.dataset.data[0].fillna(0, inplace=True)
        self.dataset.val_data[0].fillna(0, inplace=True)
        self.models = models
        self.trained_models = []
        self.explainers = explainers
        self.metrics = metrics
        self.explanations = {}
        self.ground_truth_explanations = {}

    def check_dataset(self):
        assert type(self.dataset.data) == tuple
        assert len(self.dataset.data) == 2
        assert len(self.dataset.data[0]) == len(self.dataset.data[1])

    def train_models(self):
        self.check_dataset()
        X, y = self.dataset.data[0].to_numpy(), self.dataset.data[1]
        for idx, model in enumerate(self.models):
            if model.model == "dataset":
                self.models[idx] = DatasetAsModel(self.dataset.data_class)
                self.trained_models.append(DatasetAsModel(self.dataset.data_class))
            else:
                self.trained_models.append(model.train(X, y.ravel()))
        return self.trained_models

    def generate_explanations(self, trained_model, explainer):
        X_train = self.dataset.data[0]
        X_val = self.dataset.val_data[0]
        explainer = explainer.explainer(trained_model, X_train) if explainer.name == "breakdown" else explainer.explainer(trained_model.predict, X_train)
        feature_weights = explainer.explain(X_val)
        feature_weights_train = None
        metrics_names = [m.name for m in self.metrics]
        if "roar" in metrics_names or "roar_faithfulness" in metrics_names or "roar_monotonicity" in metrics_names:
            feature_weights_train = explainer.explain(X_train)
        return feature_weights, feature_weights_train

    def generate_ground_truth_explanations(self, trained_model):
        if "shapley" not in [
            m.name for m in self.metrics
        ]:  # compute gt shapley only when necessary
            return np.zeros((self.dataset.val_data[0].shape[0], 1)), np.zeros_like(
                self.dataset.val_data[0]
            )
        if self.dataset.data_class:
            X = self.dataset.val_data[0]
            explainer = GroundTruthShap(trained_model.predict, self.dataset.data_class)

            ground_truth_expectations = []
            ground_truth_weights = []

            for x in tqdm(X.to_numpy()):
                ground_truth_expectation, ground_truth_weight = explainer.explain(x)
                ground_truth_expectations.append(ground_truth_expectation)
                ground_truth_weights.append(ground_truth_weight)

            return np.squeeze(np.array(ground_truth_expectations)), np.squeeze(np.array(ground_truth_weights))

        raise NotImplementedError(
            "Cannot use GroundTruthShap without knowing the underlying distribution. Use one of the synthetic datasets."
        )

    def evaluate_explanations(
        self, model, trained_model, metric, feature_weights, ground_truth_weights, X_train_feature_weights
    ):
        X, y = self.dataset.val_data[0], self.dataset.val_data[1]
        X_train, y_train = self.dataset.data[0], self.dataset.data[1]
        metric = metric.metric(model, trained_model, self.dataset.data_class)
        return metric.evaluate(
            X,
            y,
            feature_weights,
            ground_truth_weights,
            X_train=X_train,
            y_train=y_train,
            X_train_feature_weights=X_train_feature_weights
        )
    
    def get_metric(self, mode, model):
        if mode == "regression":
            train_X, test_X = self.dataset.data[0], self.dataset.val_data[0]
            train_y, test_y = self.dataset.data[1], self.dataset.val_data[1]
            train_score, test_score = mean_squared_error(model.predict(train_X), train_y), mean_squared_error(model.predict(test_X), test_y)
        else:
            train_X, test_X = self.dataset.data[0], self.dataset.val_data[0]
            train_y, test_y = self.dataset.data[1], self.dataset.val_data[1]
            train_score, test_score = accuracy_score(model.predict(train_X), train_y), accuracy_score(model.predict(test_X), test_y)
        return "{:.2f}".format(train_score), "{:.2f}".format(test_score)
    
    def log_model_metrics(self):
        scores = {}
        for model, trained_model in zip(self.models, self.trained_models):
            scores[model.name] = self.get_metric(self.dataset.mode, trained_model)
        return scores

    def get_results(self):
        
        if not self.trained_models:
            self.train_models()
        results = {}
        results["dataset"] = self.dataset.name
        results["dataset_kwargs"] = self.dataset.kwargs
        results["models"] = {}
        results["model_perfs"] = self.log_model_metrics()

        for model, trained_model in zip(self.models, self.trained_models):
            results["models"][model.name] = {}
            if model.name not in self.ground_truth_explanations:
                ground_truth_expectations, ground_truth_weights= self.generate_ground_truth_explanations(trained_model)
                self.ground_truth_explanations[model.name] = ground_truth_expectations, ground_truth_weights
                self.explanations[model.name] = {}
            
            for explainer in self.explainers:
                logging.info(f"Explaining {model.name} with {explainer.name}")
                if explainer.name not in self.explanations[model.name]:
                    feature_weights, feature_weights_train = self.generate_explanations(trained_model, explainer)
                    self.explanations[model.name][explainer.name] = feature_weights, feature_weights_train
                results["models"][model.name][explainer.name] = {}

                for metric in self.metrics:
                    score = self.evaluate_explanations(
                        model,
                        trained_model,
                        metric,
                        self.explanations[model.name][explainer.name][0],
                        self.ground_truth_explanations[model.name][1],
                        self.explanations[model.name][explainer.name][1]
                    )
                    results["models"][model.name][explainer.name][metric.name] = score

        return results
