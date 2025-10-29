import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys
import os
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models import CIFARImputationModel
    from datasets.cifar_missing import get_cifar_missing_loaders

    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train on CIFAR with missing pixels')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--missing_type', type=str, default='random',
                        choices=['random', 'block', 'column'],
                        help='Type of missing pixels')
    parser.add_argument('--missing_rate', type=float, default=0.5,
                        help='Rate of missing pixels (0-1)')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num_paths', type=int, default=8,
                        help='Number of paths in PI-NAIM')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Data loaders
    print(f"Loading {args.dataset} dataset...")
    train_loader, test_loader = get_cifar_missing_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        missing_type=args.missing_type,
        missing_rate=args.missing_rate,
        num_workers=args.num_workers
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Model
    model = CIFARImputationModel(
        input_channels=3,
        hidden_dim=args.hidden_dim,
        num_paths=args.num_paths
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for batch in pbar:
            incomplete = batch['incomplete'].to(device)
            mask = batch['mask'].to(device)
            complete = batch['complete'].to(device)

            optimizer.zero_grad()
            imputed, _ = model(incomplete, mask)
            loss = criterion(imputed * (1 - mask), complete * (1 - mask))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}: Average Loss = {avg_loss:.6f}')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoints/{args.dataset}_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print("Training completed!")


if __name__ == '__main__':
    main()