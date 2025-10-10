import torch
from pi_naim.utils.data import get_clinical_dataloaders
from pi_naim.models.pi_naim import PINAIM, RouteCfg

# Test data loading
print("Testing data loading...")
train_loader, val_loader, test_loader, input_dim, output_dim = get_clinical_dataloaders(
    batch_size=32, n_samples=100, n_features=12, missing_rate=0.3, seed=42
)

# Test one batch
batch = next(iter(train_loader))
print(f"Batch shapes: X={batch['X_true'].shape}, y={batch['y'].shape}, mask={batch['mask'].shape}")

# Test model creation
print("Testing model creation...")
cfg = RouteCfg()
model = PINAIM(input_dim, output_dim, hidden_dim=128, embed_dim=64, cfg=cfg)
print(f"Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward pass
print("Testing forward pass...")
with torch.no_grad():
    outputs = model(batch['X_true'], batch['mask'])
    print("Forward pass successful!")
    print(f"Output keys: {list(outputs.keys())}")

print("All tests passed!")