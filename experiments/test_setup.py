# experiments/test_setup.py
import torch
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test if all required imports work"""
    try:
        from pi_naim.utils.data import get_clinical_dataloaders
        from pi_naim.models.pi_naim import PINAIM, RouteCfg
        from pi_naim.utils.trainers import PINAIMTrainer
        from pi_naim.utils.curriculum import CurriculumCfg

        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_data_loading():
    """Test if data loading works"""
    try:
        train_loader, val_loader, test_loader, input_dim, output_dim = get_clinical_dataloaders(
            batch_size=8, n_samples=100, n_features=6, missing_rate=0.2, seed=42
        )

        # Test one batch
        batch = next(iter(train_loader))
        assert 'X_true' in batch
        assert 'y' in batch
        assert 'mask' in batch

        print(f"✅ Data loading successful!")
        print(f"   Input dimension: {input_dim}")
        print(f"   Output dimension: {output_dim}")
        print(f"   Batch shapes: X={batch['X_true'].shape}, y={batch['y'].shape}, mask={batch['mask'].shape}")
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False


def test_model_creation():
    """Test if model creation works"""
    try:
        cfg = RouteCfg()
        model = PINAIM(6, 2, hidden_dim=32, embed_dim=16, cfg=cfg)

        print(f"✅ Model creation successful!")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


def main():
    print("Testing PI-NAIM setup...")
    print("=" * 50)

    # Test imports
    if not test_imports():
        return False

    # Test data loading
    if not test_data_loading():
        return False

    # Test model creation
    if not test_model_creation():
        return False

    print("=" * 50)
    print("✅ All tests passed! Setup is correct.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)