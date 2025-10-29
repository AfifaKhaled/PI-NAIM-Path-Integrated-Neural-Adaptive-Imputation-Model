import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import os


def check_dependencies():
    """Check and report missing dependencies with clear instructions"""
    print("Checking dependencies...")
    print("=" * 50)

    missing = []

    # Check torch
    try:
        import torch
        # Check if torch has the expected attributes (indicates proper installation)
        if hasattr(torch, '__version__'):
            print(f"✓ PyTorch {torch.__version__}")
        else:
            print("✗ PyTorch is corrupted (no __version__ attribute)")
            missing.append("torch")
    except ImportError:
        print("✗ PyTorch not installed")
        missing.append("torch")

    # Check torchvision
    try:
        import torchvision
        if hasattr(torchvision, '__version__'):
            print(f"✓ TorchVision {torchvision.__version__}")
        else:
            print("✗ TorchVision is corrupted")
            missing.append("torchvision")
    except ImportError:
        print("✗ TorchVision not installed")
        missing.append("torchvision")

    # Check scikit-image
    try:
        import skimage
        print("✓ scikit-image")
    except ImportError:
        print("✗ scikit-image not installed")
        missing.append("scikit-image")

    # Check tqdm
    try:
        import tqdm
        print("✓ tqdm")
    except ImportError:
        print("✗ tqdm not installed")
        missing.append("tqdm")

    # Check tensorboard
    try:
        import tensorboard
        print("✓ tensorboard")
    except ImportError:
        print("✗ tensorboard not installed")
        missing.append("tensorboard")

    # Check PIL/Pillow
    try:
        from PIL import Image
        print("✓ Pillow")
    except ImportError:
        print("✗ Pillow not installed")
        missing.append("pillow")

    print("=" * 50)

    if missing:
        print(f"\n❌ Missing or corrupted dependencies: {', '.join(missing)}")
        print("\nTo fix this, please run these commands:")
        print("\n1. First, remove the current environment:")
        print("   conda deactivate")
        print("   conda remove -n my_env --all")
        print("\n2. Create a new environment:")
        print("   conda create -n naim_env python=3.9 -y")
        print("   conda activate naim_env")
        print("\n3. Install dependencies:")
        print("   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch -y")
        print("   pip install scikit-image tqdm tensorboard Pillow numpy matplotlib")
        return False

    print("\n✅ All dependencies are installed correctly!")

    # Additional system info
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    return True


# Check dependencies first
if not check_dependencies():
    sys.exit(1)

# Now import the rest of the modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from datasets.cifar_missing import get_cifar_missing_loaders
    from models.models_cifar import CIFARImputationModel, CIFARClassifier
except ImportError as e:
    print(f"❌ Project structure error: {e}")
    print("\nPlease ensure your project has this structure:")
    print("PI-NAIM-Path-Integrated-Neural-Adaptive-Imputation-Model-main/")
    print("├── datasets/")
    print("│   ├── __init__.py")
    print("│   └── cifar_missing.py")
    print("├── models/")
    print("│   ├── __init__.py")
    print("│   └── models_cifar.py")
    print("└── scripts/")
    print("    └── train_cifar.py")
    sys.exit(1)


def train_cifar(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Data loaders
    print("Loading CIFAR dataset...")
    train_loader, test_loader = get_cifar_missing_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        missing_type=args.missing_type,
        missing_rate=args.missing_rate
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Models
    print("Initializing models...")
    imputation_model = CIFARImputationModel(
        input_channels=3,
        hidden_dim=args.hidden_dim,
        num_paths=args.num_paths
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in imputation_model.parameters()):,}")

    # Optimizer
    optimizer = optim.Adam(imputation_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # TensorBoard
    writer = SummaryWriter(f'logs/cifar_{args.dataset}_{args.missing_type}')

    print("Starting training...")
    # Training loop
    for epoch in range(args.epochs):
        imputation_model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            incomplete = batch['incomplete'].to(device)
            mask = batch['mask'].to(device)
            complete = batch['complete'].to(device)

            optimizer.zero_grad()

            # Forward pass
            imputed, reconstructed = imputation_model(incomplete, mask)

            # Loss on missing regions
            loss = criterion(imputed * (1 - mask), complete * (1 - mask))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Validation
        imputation_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                incomplete = batch['incomplete'].to(device)
                mask = batch['mask'].to(device)
                complete = batch['complete'].to(device)

                imputed, _ = imputation_model(incomplete, mask)
                loss = criterion(imputed * (1 - mask), complete * (1 - mask))
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)

        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # TensorBoard logging
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f'checkpoints/cifar_{args.dataset}_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': imputation_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PI-NAIM on CIFAR with missing pixels')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--missing_type', type=str, default='random',
                        choices=['random', 'block', 'column'])
    parser.add_argument('--missing_rate', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_paths', type=int, default=8)
    parser.add_argument('--save_interval', type=int, default=10)

    args = parser.parse_args()

    # Print configuration
    print("Configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print()

    train_cifar(args)