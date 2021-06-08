import pandas as pd

class CustomDataset:
    def __init__(self, num_train_samples, num_val_samples, num_classes=None):
        self.num_train_samples = num_train_samples or 1000
        self.num_val_samples = num_val_samples or 100
        self.num_classes = num_classes or 1

    def get_dataset(self, num_samples=None):
        X, y = self.generate(n_sample=num_samples or self.num_train_samples)
        return pd.DataFrame(X), y