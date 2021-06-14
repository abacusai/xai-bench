import pandas as pd
import numpy as np
from synthetic_datasets import GaussianLinearRegression, GaussianLinearBinary
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from scipy.spatial.distance import jensenshannon as jsd
n_bins = [0.1*x for x in range(-60,60,1)]
df = pd.read_csv('data/winequality-white.csv', sep=';')
features = list(df.columns)[0:-1]
print(features)
X = df.drop('quality', axis=1)
X = preprocessing.StandardScaler().fit(X).transform(X)
y = df['quality']
#print(y)
knn = KNeighborsRegressor(n_neighbors=1)
#knn = KNeighborsClassifier()
knn.fit(X,y)
mse = np.mean((knn.predict(X) - y)**2)
print('MSE: ',mse)
mean = np.mean(X, axis=0)
cov = np.cov(X, rowvar=False)

data_generator = GaussianLinearRegression(mu=mean, dim=len(mean), noise=0.01, 
                                            sigma=cov, weight=np.ones(len(mean)))

synthetic_samples, _ = data_generator.get_dataset(num_samples=10000)
y_synthetic = knn.predict(synthetic_samples)
#print(np.array(synthetic_samples))
#plt.figure(figsize = (10,10))
#sns.heatmap(cov, annot = True, cmap = 'coolwarm')
fig, ax = plt.subplots()
im = ax.imshow(cov)

# We want to show all ticks...
ax.set_xticks(np.arange(len(features)))
ax.set_yticks(np.arange(len(features)))
# ... and label them with the respective list entries
ax.set_xticklabels(features)
ax.set_yticklabels(features)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(features)):
    for j in range(len(features)):
        text = ax.text(j, i, "{:.2f}".format(cov[i, j]),
                       ha="center", va="center", color="w")

ax.set_title("Empirical feature covariance matrix")
fig.tight_layout()
plt.show()

#plt.savefig('heat.pdf',dpi=fig.dpi)

plt.figure(figsize = (10,10))
fig, axs = plt.subplots(2,X.shape[1])

n_real = X.shape[0]
n_synth = np.array(synthetic_samples).shape[0]

KL = []
KL_2 = []
JSD = []
for feature in range(X.shape[1]):
    counts_real,bins = np.histogram (X[:,feature],bins=n_bins)
    counts_synth,bins = np.histogram (np.array(synthetic_samples)[:,feature],bins=n_bins)
    
    prob_real = counts_real/n_real
    prob_synth = counts_synth/n_synth
    

    axs[0,feature].hist(X[:,feature],bins=n_bins) #= sns.displot(data=X[:,feature])
    axs[1,feature].hist(np.array(synthetic_samples)[:,feature],bins=n_bins)
    
    js = jsd(prob_real,prob_synth)
    JSD.append(js)

print("Marginal Feature JSD: ",JSD)
print("Mean Marginal JSD: ",np.mean(JSD))

#fig, axs = plt.subplots(1,2)
#n_bins_y = [x for x in range(0,10,1)]
#axs[0].hist(y,bins=n_bins_y)
#axs[1].hist(y_synthetic,bins=n_bins_y)

prob_real = counts_real/n_real
prob_synth = counts_synth/n_synth

js = jsd(prob_real,prob_synth)

print("target JSD:", js)
