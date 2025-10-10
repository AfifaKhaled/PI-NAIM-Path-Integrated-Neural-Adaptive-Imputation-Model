# experiments/baseline.py
import torch
import torch.nn as nn
import numpy as np
from pi_naim.utils.data import get_clinical_dataloaders


class SimpleBaseline(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x, m):
        # Simple mean imputation
        col_means = (x * m).sum(dim=0) / m.sum(dim=0).clamp(min=1)
        x_imp = x * m + (1 - m) * col_means.unsqueeze(0)
        return self.net(x_imp)


def test_baseline():
    train_loader, val_loader, test_loader, input_dim, output_dim = get_clinical_dataloaders(
        batch_size=32, n_samples=1000, n_features=8, missing_rate=0.2, seed=42
    )

    model = SimpleBaseline(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train simple baseline
    model.train()
    for epoch in range(20):
        total_loss = 0
        for batch in train_loader:
            x, y, mask = batch["X_true"], batch["y"], batch["mask"]
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss {total_loss / len(train_loader):.4f}")

    # Evaluate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y, mask = batch["X_true"], batch["y"], batch["mask"]
            logits = model(x, mask)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Baseline Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    test_baseline()