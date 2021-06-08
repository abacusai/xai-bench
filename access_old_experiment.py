import pickle
from os import walk
import os

from src import datasets, model, explainer, metric, experiments, parse_utils

res_dir = "results/regression/exp-gaussian-dim=100-1/"
exp_path = f"{res_dir}/checkpoints/"

_, _, filenames = next(walk(exp_path))

for filename in filenames:
    with open(os.path.join(exp_path, filename), 'rb') as f:
        experiment = pickle.load(f)
        experiment.metrics = []
        experiment.metrics.append(metric.Metric('faithfulness', version="inc"))
        # experiment.metrics.append(metric.Metric('roar_faithfulness', version="inc"))
        experiment.metrics.append(metric.Metric('monotonicity', version="dec"))
        # experiment.metrics.append(metric.Metric('roar_monotonicity', version="dec"))
        res = experiment.get_results()
        parse_utils.save_results_csv(res, res_dir+"/more_csv/")
