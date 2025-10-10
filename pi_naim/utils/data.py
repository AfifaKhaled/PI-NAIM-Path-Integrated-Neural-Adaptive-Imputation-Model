# pi_naim/utils/data.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
from .mimic_data import MIMICDataLoader, get_mimic_dataloaders


class ClinicalDataset(Dataset):
    def __init__(self, features, labels, mask=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.float32) if mask is not None else torch.ones_like(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "X_true": self.features[idx],
            "y": self.labels[idx],
            "mask": self.mask[idx]
        }


def augment_batch(batch, noise_level=0.03, feature_dropout=0.1):
    """Enhanced data augmentation for better regularization"""
    X, y, mask = batch["X_true"], batch["y"], batch["mask"]

    # Add Gaussian noise only to observed values
    noise = torch.randn_like(X) * noise_level
    X_aug = X + noise * mask

    # Random feature masking for additional regularization
    if feature_dropout > 0:
        feature_mask = (torch.rand(X.shape[1]) > feature_dropout).float().to(X.device)
        X_aug = X_aug * feature_mask.unsqueeze(0)
        # Update mask to reflect feature dropout
        mask = mask * feature_mask.unsqueeze(0)

    return {"X_true": X_aug, "y": y, "mask": mask}


def create_clinical_dataset(n_samples=2000, n_features=15, missing_rate=0.3, seed=42):
    """
    Create a realistic clinical dataset with improved feature correlations
    and more complex missingness patterns
    """
    np.random.seed(seed)

    # Create more realistic correlated features
    cov_matrix = np.eye(n_features)
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                # Features are more correlated if they're in similar categories
                feature_type_i = i % 3  # 3 types of features
                feature_type_j = j % 3
                if feature_type_i == feature_type_j:
                    cov_matrix[i, j] = 0.7  # High correlation within same type
                else:
                    cov_matrix[i, j] = 0.3  # Lower correlation across types

    # Generate features from multivariate normal distribution
    X = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=cov_matrix,
        size=n_samples
    )

    # Create more realistic target variable with non-linear relationships
    vital_signs = list(range(0, n_features, 3))
    lab_results = list(range(1, n_features, 3))
    treatment_vars = list(range(2, n_features, 3))

    # Non-linear relationships with different feature types
    coefficients = np.zeros(n_features)
    coefficients[vital_signs] = 0.8  # Most important
    coefficients[lab_results] = 0.5  # Medium importance
    coefficients[treatment_vars] = 0.3  # Least important

    # Add non-linear terms
    X_squared = X ** 2
    interaction_terms = X[:, vital_signs].mean(axis=1, keepdims=True) * X[:, lab_results].mean(axis=1, keepdims=True)

    logits = (np.dot(X, coefficients) +
              0.1 * np.dot(X_squared, coefficients) +
              0.2 * interaction_terms.flatten() +
              np.random.randn(n_samples) * 0.2)

    y_proba = 1 / (1 + np.exp(-logits))
    y = (y_proba > 0.5).astype(int)

    # Ensure balanced classes
    positive_ratio = np.mean(y)
    if positive_ratio < 0.4 or positive_ratio > 0.6:
        # Adjust threshold to balance classes
        threshold = np.percentile(y_proba, 50)  # Median split
        y = (y_proba > threshold).astype(int)

    # Create missing mask with more realistic patterns
    mask = np.ones((n_samples, n_features))

    for i in range(n_samples):
        missing_probs = np.full(n_features, missing_rate)

        # Increase missingness for extreme values (MNAR)
        for j in range(n_features):
            if abs(X[i, j]) > 2.0:  # Very extreme values
                missing_probs[j] = min(missing_rate + 0.4, 0.9)
            elif abs(X[i, j]) > 1.5:  # Extreme values
                missing_probs[j] = min(missing_rate + 0.2, 0.7)

        # Increase missingness based on outcome (MAR)
        if y[i] == 1:  # Positive cases
            missing_probs[lab_results] = min(missing_rate + 0.2, 0.7)  # Increased effect
            missing_probs[treatment_vars] = min(missing_rate + 0.15, 0.6)

        # Increase missingness for certain feature types
        missing_probs[treatment_vars] = min(missing_rate + 0.15, 0.55)

        # Add random variation
        missing_probs = missing_probs * np.random.uniform(0.9, 1.1, n_features)
        missing_probs = np.clip(missing_probs, 0.1, 0.9)

        mask[i] = (np.random.rand(n_features) > missing_probs).astype(float)

    return X, y, mask


def get_clinical_dataloaders(batch_size=32, n_samples=2000, n_features=15,
                             missing_rate=0.3, seed=42, augment=False):
    X, y, mask = create_clinical_dataset(
        n_samples=n_samples,
        n_features=n_features,
        missing_rate=missing_rate,
        seed=seed
    )

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    dataset = ClinicalDataset(X, y, mask)

    # Train/validation/test split (70%/15%/15%)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Add data augmentation to training loader
    if augment:
        def augmented_collate(batch):
            batch = torch.utils.data.default_collate(batch)
            return augment_batch(batch)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=augmented_collate)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, X.shape[1], len(np.unique(y))


def get_admissions_dataloaders(batch_size=32, missing_rate=0.2, seed=42):
    """
    Legacy function for backward compatibility
    """
    return get_clinical_dataloaders(
        batch_size=batch_size,
        n_samples=800,
        n_features=8,
        missing_rate=missing_rate,
        seed=seed
    )
def get_mimic_dataloaders(data_dir, batch_size=32, missing_rate=0.3,
                         target_condition="mortality", seed=42):
    """
    Get MIMIC-III dataloaders for training
    """
    loader = MIMICDataLoader(data_dir)
    return loader.create_mimic_dataloaders(
        batch_size=batch_size,
        missing_rate=missing_rate,
        target_condition=target_condition,
        seed=seed
    )