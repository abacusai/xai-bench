from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

valid_models = {
    "regression": {
        "dataset": lambda : "dataset",
        "lr": LinearRegression,
        "mlp": MLPRegressor,
        "dtree": DecisionTreeRegressor, 
    },
    "classification": {
        "dataset": lambda : "dataset",
        "lr": LinearRegression,
        "mlp": MLPClassifier,
        "dtree": DecisionTreeClassifier,
    },
}


class Model:
    def __init__(self, name, mode, **kwargs):
        if name not in valid_models[mode].keys():
            raise NotImplementedError(
                f"This model is not supported at the moment. Models supported are: {list(valid_models[mode].keys())}"
            )
        self.name = name
        self.mode = mode
        self.model = valid_models[mode][name](**kwargs)
        if self.model == "dataset":
            return
        self.predict = self.model.predict
        if self.model.fit:
            self.train = self.model.fit
