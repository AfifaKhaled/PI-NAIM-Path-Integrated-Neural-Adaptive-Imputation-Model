# experiments/mimic_experiment.py

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_naim.utils.mimic_data import MIMICDataLoader
from pi_naim.models.pi_naim import PINAIM, RouteCfg
from pi_naim.utils.trainers import PINAIMTrainer
from pi_naim.utils.curriculum import CurriculumCfg


def setup_device(device_preference):
    """Setup device for training"""
    if device_preference == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif device_preference == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def run_mimic_experiment():
    parser = argparse.ArgumentParser(description='PI-NAIM MIMIC-III Experiment')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to MIMIC-III CSV files')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--missing_rate', type=float, default=0.3)
    parser.add_argument('--target', type=str, default='mortality',
                        choices=['mortality', 'sepsis', 'specific_icd'])
    parser.add_argument('--output_dir', type=str, default='mimic_results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = setup_device(args.device)

    # Load MIMIC data
    print("Loading MIMIC-III data...")
    mimic_loader = MIMICDataLoader(args.data_dir)

    train_loader, val_loader, test_loader, input_dim, output_dim = mimic_loader.create_mimic_dataloaders(
        batch_size=args.batch_size,
        missing_rate=args.missing_rate,
        target_condition=args.target,
        seed=42
    )

    # Create model
    cfg = RouteCfg()
    model = PINAIM(input_dim, output_dim, hidden_dim=64, embed_dim=32, cfg=cfg)

    # Setup optimizer and trainer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    curriculum_cfg = CurriculumCfg(epochs=args.epochs)

    # FIXED: Use the actual device object instead of string 'auto'
    trainer = PINAIMTrainer(model, optimizer, device=device, curriculum_cfg=curriculum_cfg)

    # Train
    print("Starting training...")
    trainer.train(
        train_loader, val_loader, epochs=args.epochs,
        checkpoint_path=os.path.join(args.output_dir, 'best_model.pth')
    )

    # Evaluate
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test Results: {test_metrics}")

    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    trainer.save_training_report(results_path)

    print(f"Experiment completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    run_mimic_experiment()