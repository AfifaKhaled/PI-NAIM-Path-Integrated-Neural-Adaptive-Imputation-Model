# pi_naim/data.py

import numpy as np
import torch
from torch.utils.data import Dataset

class AdmissionsDataset(Dataset):
    """
    A simple dataset wrapper for admissions data.
    Stores features X, labels y, and missingness mask.
    """
    def __init__(self, X, y, mask):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


def load_admissions(missing_rate=0.2, seed=42):
    """
    Generates synthetic admissions dataset with missing values.
    Replace this later with real admissions.csv if available.
    """
    rng = np.random.RandomState(seed)

    n_samples = 1000
    n_features = 6
    n_classes = 2

    # Generate random features and labels
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, n_classes, size=n_samples)

    # Missing mask (1 = observed, 0 = missing)
    mask = rng.rand(n_samples, n_features) > missing_rate

    return X, y, mask
