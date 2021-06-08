import pandas as pd
import numpy as np
from synthetic_datasets import GaussianLinearRegression, GaussianLinearBinary
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from src import explainer
from sklearn.metrics import mean_squared_error

explainers = [ "shap",
    #"shapr",
    #"kernelshap",
    #"brutekernelshap",
    #"random",
    "lime",
    "maple",
    "l2x",
]
model = 'tree'
df = pd.read_csv('data/winequality-white.csv', sep=';')

X = df.drop('quality', axis=1)
X = preprocessing.StandardScaler().fit(X).transform(X)
y = df['quality']

X, X_val, y, y_val = train_test_split(X, y, test_size=0.05, random_state=7)
#print(y)
knn = KNeighborsRegressor(n_neighbors=1)
#knn = KNeighborsClassifier()
knn.fit(X,y)
# mse = np.mean((knn.predict(X) - y)**2)
# print('MSE: ',mse)
mean = np.mean(X, axis=0)
cov = np.cov(X, rowvar=False)

data_generator = GaussianLinearRegression(mu=mean, dim=len(mean), noise=0.01, 
                                            sigma=cov, weight=np.ones(len(mean)))

synthetic_samples, _ = data_generator.get_dataset(num_samples=len(X))
y_synthetic = knn.predict(synthetic_samples)

X, X_val = pd.DataFrame(X), pd.DataFrame(X_val)
if model == "mlp":
    print('MLP')
    model_real = MLPRegressor().fit(X, y)
    model_syn = MLPRegressor().fit(synthetic_samples, y_synthetic)
elif model == "tree":
    print('tree')
    model_real = DecisionTreeRegressor().fit(X, y)
    model_syn = DecisionTreeRegressor().fit(synthetic_samples, y_synthetic)
elif model == "lr":
    print('linear regression')
    model_real = LinearRegression().fit(X, y)
    model_syn = LinearRegression().fit(synthetic_samples, y_synthetic)
else:
    print("Model not supported")

def get_real_syn_explanations_mse(real, syn):
    return mean_squared_error(real, syn)

for explainer_name in explainers:
    explainer_real = explainer.Explainer(explainer_name).explainer(model_real.predict, X)
    explainer_syn = explainer.Explainer(explainer_name).explainer(model_syn.predict, synthetic_samples)
    feature_weights_real = explainer_real.explain(X_val)
    feature_weights_syn = explainer_syn.explain(X_val)
    mse = get_real_syn_explanations_mse(feature_weights_real, feature_weights_syn)
    print(f"{explainer_name} mse is: {mse}")
    print("model: ", model)

# shap mse is: 0.0396764547060464
# brutekernelshap mse is: 0.05015404817057441
# random mse is: 1.8889475270676486
# lime mse is: 0.03713335058317393
# maple mse is: 0.06605651721865091
# l2x mse is: 0.0008952742128657123