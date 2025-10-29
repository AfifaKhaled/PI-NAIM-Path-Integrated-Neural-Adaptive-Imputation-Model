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

# Import CIFAR-specific modules
from models import CIFARImputationModel
from datasets.cifar_missing import get_cifar_missing_loaders


def train_cifar(args):
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
    print("Initializing model...")
    model = CIFARImputationModel(
        input_channels=3,
        hidden_dim=args.hidden_dim,
        num_paths=args.num_paths
    ).to(device)

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # TensorBoard
    writer = SummaryWriter(f'logs/{args.dataset}_{args.missing_type}_{args.missing_rate}')

    print("Starting training...")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            incomplete = batch['incomplete'].to(device)
            mask = batch['mask'].to(device)
            complete = batch['complete'].to(device)

            optimizer.zero_grad()

            # Forward pass
            imputed, _ = model(incomplete, mask)

            # Loss on missing regions only
            loss = criterion(imputed * (1 - mask), complete * (1 - mask))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                incomplete = batch['incomplete'].to(device)
                mask = batch['mask'].to(device)
                complete = batch['complete'].to(device)

                imputed, _ = model(incomplete, mask)
                loss = criterion(imputed * (1 - mask), complete * (1 - mask))
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)

        print(f'Epoch {epoch + 1}/{args.epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {avg_val_loss:.6f}')

        # TensorBoard logging
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = f'checkpoints/{args.dataset}_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PI-NAIM on CIFAR with missing pixels')

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
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Print configuration
    print("=" * 50)
    print("CIFAR Training Configuration")
    print("=" * 50)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 50)
    print()

    train_cifar(args)