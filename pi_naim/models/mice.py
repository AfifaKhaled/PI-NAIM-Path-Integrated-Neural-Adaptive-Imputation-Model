# pi_naim/models/mice.py
import torch
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings


class BayesianMICE:
    def __init__(self, max_iter=5, random_state=42, bootstrap_samples=3):
        self.max_iter = max_iter
        self.random_state = random_state
        self.bootstrap_samples = bootstrap_samples
        self.imputers = []
        self.estimator = RandomForestRegressor(n_estimators=50, random_state=random_state)

    def fit_transform(self, X_np, mask_np):
        """Bayesian MICE with bootstrap uncertainty"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            imputer = IterativeImputer(
                estimator=self.estimator,
                max_iter=self.max_iter,
                random_state=self.random_state,
                sample_posterior=False
            )

            # Multiple imputation for uncertainty
            imputations = []
            for i in range(self.bootstrap_samples):
                # Bootstrap sampling - ensure we maintain the same shape
                idx = np.random.choice(len(X_np), size=len(X_np), replace=True)
                X_boot = X_np[idx].copy()
                mask_boot = mask_np[idx].copy()

                # Impute
                X_imp = imputer.fit_transform(X_boot)

                # Ensure all imputations have the same shape
                if X_imp.shape[1] != X_np.shape[1]:
                    # If shape mismatch, pad with zeros or use original shape
                    if X_imp.shape[1] < X_np.shape[1]:
                        # Pad with zeros
                        pad_width = ((0, 0), (0, X_np.shape[1] - X_imp.shape[1]))
                        X_imp = np.pad(X_imp, pad_width, mode='constant')
                    else:
                        # Truncate to original shape
                        X_imp = X_imp[:, :X_np.shape[1]]

                imputations.append(X_imp)

            # Stack imputations - now they should all have the same shape
            imputations = np.stack(imputations)
            mean_imp = imputations.mean(axis=0)
            var_imp = imputations.var(axis=0)

            return mean_imp, var_imp


def run_mice_impute(X, mask, max_iter=5, bootstrap=3):
    """Run MICE imputation with proper tensor handling"""
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
        mask_np = mask.cpu().numpy()
    else:
        X_np = X.copy()
        mask_np = mask.copy()

    # Apply mask
    X_masked = X_np.copy()
    X_masked[mask_np == 0] = np.nan

    # Run Bayesian MICE
    mice = BayesianMICE(max_iter=max_iter, bootstrap_samples=bootstrap)
    mean_imp, var_imp = mice.fit_transform(X_masked, mask_np)

    return mean_imp, var_imp