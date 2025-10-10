# experiments/Main.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import sys
import os
import json
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_naim.utils.data import get_clinical_dataloaders
from pi_naim.models.pi_naim import PINAIM, RouteCfg
from pi_naim.utils.trainers import PINAIMTrainer
from pi_naim.utils.curriculum import CurriculumCfg
from pi_naim.utils.metrics import rmse, auroc
from pi_naim.utils.mimic_data import MIMICDataLoader, get_mimic_dataloaders


def setup_experiment():
    """Setup experiment configuration matching paper parameters"""
    parser = argparse.ArgumentParser(description='PI-NAIM Training Experiment - Optimized Version')

    # Optimized parameters for better performance
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=2000, help='Number of samples in dataset')  # Increased
    parser.add_argument('--n_features', type=int, default=15, help='Number of features')
    parser.add_argument('--missing_rate', type=float, default=0.3, help='Missing data rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for generator')
    parser.add_argument('--lr_critic', type=float, default=0.0001, help='Learning rate for critic')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--save_dir', type=str, default='results_optimized', help='Directory to save results')
    parser.add_argument('--dataset', type=str, default='clinical',
                        choices=['clinical', 'synthetic', 'mimic'],
                        help='Dataset type')
    parser.add_argument('--mimic_dir', type=str, default=None,
                        help='Path to MIMIC-III CSV files (required for mimic dataset)')
    parser.add_argument('--target_condition', type=str, default='mortality',
                        choices=['mortality', 'sepsis'],
                        help='Target condition for MIMIC dataset')

    # Enhanced regularization parameters
    parser.add_argument('--route_threshold', type=float, default=0.15,
                        help='Lower threshold for more GAIN routing')  # Lowered
    parser.add_argument('--temporal', type=bool, default=True, help='Enable temporal attention')
    parser.add_argument('--n_bootstrap', type=int, default=3, help='Number of bootstrap samples for MICE')
    parser.add_argument('--gp_lambda', type=float, default=10.0, help='Gradient penalty coefficient')
    parser.add_argument('--alpha_rec', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--n_critic', type=int, default=3, help='Critic iterations per generator update')
    parser.add_argument('--attention_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--mice_max_iter', type=int, default=5, help='MICE maximum iterations')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience (0 to disable)')  # Increased
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for regularization')
    parser.add_argument('--gain_dropout', type=float, default=0.3, help='Dropout for GAIN path')
    parser.add_argument('--task_dropout', type=float, default=0.2, help='Dropout for task head')
    parser.add_argument('--use_cosine_lr', type=bool, default=True, help='Use cosine learning rate schedule')  # New

    return parser.parse_args()


def create_save_directory(save_dir):
    """Create directory for saving results"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {os.path.abspath(save_dir)}")
        return save_dir
    except Exception as e:
        print(f"Error creating directory {save_dir}: {e}")
        # Fallback to current directory
        fallback_dir = "results_optimized"
        os.makedirs(fallback_dir, exist_ok=True)
        print(f"Using fallback directory: {os.path.abspath(fallback_dir)}")
        return fallback_dir


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


def main():
    try:
        # Parse command line arguments
        args = setup_experiment()

        # Create save directory
        save_dir = create_save_directory(args.save_dir)

        # Set random seeds for reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Setup device
        device = setup_device(args.device)

        print("=" * 60)
        print("           PI-NAIM OPTIMIZED TRAINING EXPERIMENT")
        print("=" * 60)
        print(f"Random seed: {args.seed}")
        print(f"Device: {device}")
        print(f"Save directory: {save_dir}")
        print(f"Early stopping: {'Disabled' if args.early_stopping == 0 else f'Patience {args.early_stopping}'}")
        print(f"Weight decay: {args.weight_decay}")
        print(f"GAIN dropout: {args.gain_dropout}")
        print(f"Route threshold: {args.route_threshold}")
        print(f"Dataset: {args.dataset}")
        if args.dataset == 'mimic':
            print(f"Target condition: {args.target_condition}")
        else:
            print(f"Dataset size: {args.n_samples}")
        print("=" * 60)

        # Load dataset based on type
        print("Loading dataset...")
        if args.dataset == 'mimic':
            if args.mimic_dir is None:
                raise ValueError("--mimic_dir is required for mimic dataset")

            print(f"Loading MIMIC-III data from: {args.mimic_dir}")
            train_loader, val_loader, test_loader, input_dim, output_dim = get_mimic_dataloaders(
                data_dir=args.mimic_dir,
                batch_size=args.batch_size,
                missing_rate=args.missing_rate,
                target_condition=args.target_condition,
                seed=args.seed
            )
        else:
            # Use clinical or synthetic data
            train_loader, val_loader, test_loader, input_dim, output_dim = get_clinical_dataloaders(
                batch_size=args.batch_size,
                n_samples=args.n_samples,
                n_features=args.n_features,
                missing_rate=args.missing_rate,
                seed=args.seed
            )

        print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")

        # Enhanced model configuration with better regularization
        cfg = RouteCfg(
            threshold=args.route_threshold,  # Use the new lower threshold
            temporal=args.temporal,
            n_bootstrap=args.n_bootstrap,
            gp_lambda=args.gp_lambda,
            alpha_rec=args.alpha_rec,
            gain_hidden=args.hidden_dim,
            gain_dropout=args.gain_dropout,
            mice_max_iter=args.mice_max_iter,
            learnable_threshold=True,
            n_critic=args.n_critic,
            attention_heads=args.attention_heads
        )

        # Create model with compatible dimensions
        print("Initializing PI-NAIM model...")

        # Ensure embed_dim is divisible by attention_heads
        embed_dim = args.embed_dim
        if embed_dim % cfg.attention_heads != 0:
            embed_dim = embed_dim + (cfg.attention_heads - embed_dim % cfg.attention_heads)
            print(f"Adjusted embed_dim to {embed_dim} for attention head compatibility")

        model = PINAIM(input_dim, output_dim,
                       hidden_dim=args.hidden_dim,
                       embed_dim=embed_dim,
                       cfg=cfg,
                       task_dropout=args.task_dropout)
        model = model.to(device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Setup separate optimizers with weight decay
        optimizer = optim.Adam(
            [p for n, p in model.named_parameters() if 'critic' not in n],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.5, 0.9)
        )

        # Enhanced learning rate scheduling
        if args.use_cosine_lr:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=int(args.epochs * 0.7),  # 70% of epochs
                eta_min=1e-6,
                last_epoch=-1
            )
            print("Using Cosine Annealing LR scheduler")
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=1e-6
            )
            print("Using ReduceLROnPlateau scheduler")

        # Full curriculum matching paper
        curriculum_cfg = CurriculumCfg(
            epochs=args.epochs,
            mcar_frac=0.3,  # 30% MCAR
            mar_frac=0.4,  # 40% MAR
            mnar_frac=0.3  # 30% MNAR
        )

        # Trainer with adversarial training
        trainer = PINAIMTrainer(
            model,
            optimizer,
            device,
            curriculum_cfg,
            lr_critic=args.lr_critic
        )

        # Train
        print(f"\nStarting training for {args.epochs} epochs...")
        checkpoint_path = os.path.join(save_dir, 'best_pinaim_model.pth')

        # Calculate estimated training time
        if torch.cuda.is_available():
            est_time_minutes = args.epochs * 0.3
        else:
            est_time_minutes = args.epochs * 2.5  # Slightly increased for larger dataset

        print(f"Estimated training time: {est_time_minutes / 60:.1f} hours")
        print("Press Ctrl+C to interrupt training and save current progress")

        history = trainer.train(
            train_loader, val_loader, epochs=args.epochs,
            scheduler=scheduler,
            early_stopping_patience=args.early_stopping if args.early_stopping > 0 else args.epochs + 1,
            checkpoint_path=checkpoint_path
        )

        # Get training summary
        summary = trainer.get_training_summary()
        print(f"\n=== TRAINING SUMMARY ===")
        print(f"Best AUROC: {summary['best_auroc']:.4f} at epoch {summary['best_epoch']}")
        print(f"Best Accuracy: {summary['best_accuracy']:.4f}")
        print(f"Final Train Loss: {summary['final_train_loss']:.4f}")
        print(f"Total Epochs Completed: {summary['num_epochs']}")

        # Final evaluation on test set with uncertainty
        print("\nEvaluating on test set with uncertainty quantification...")
        final_metrics = trainer.evaluate(test_loader, return_uncertainty=True)

        print(f"\n=== FINAL TEST RESULTS ===")
        print(f"AUROC: {final_metrics['auroc']:.4f}")
        print(f"Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"RMSE: {final_metrics['rmse']:.4f}")
        print(f"Uncertainty: {final_metrics.get('uncertainty', 0):.4f}")
        print(f"Routing to GAIN: {final_metrics.get('routing_to_gain', 0):.3f}")
        print(f"Avg Missing Rate: {final_metrics.get('missing_rate', 0):.3f}")

        # Save final model
        final_model_path = os.path.join(save_dir, 'final_pinaim_model.pth')
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'test_metrics': final_metrics,
                'training_summary': summary,
                'training_history': history,
            }, final_model_path)
            print(f"\nFinal model saved to '{final_model_path}'")
        except Exception as e:
            print(f"Error saving final model: {e}")

        # Get detailed routing statistics
        print("\nAnalyzing routing statistics...")
        routing_stats = trainer.get_routing_statistics(test_loader)
        if routing_stats:
            print(f"Routing Statistics:")
            print(f"  Avg Route to GAIN: {routing_stats['avg_route_to_gain']:.3f}")
            print(f"  % Samples to GAIN: {routing_stats['percent_routed_to_gain']:.1f}%")
            print(f"  Avg Missing Rate: {routing_stats['avg_missing_rate']:.3f}")
            print(f"  Routing Correlation: {routing_stats['routing_correlation']:.3f}")

        # Save comprehensive training report
        report_path = os.path.join(save_dir, 'training_report.json')
        trainer.save_training_report(report_path)

        # Print performance summary
        trainer.print_performance_summary()

        # Save training history plot
        plot_path = os.path.join(save_dir, 'training_history.png')
        trainer.plot_training_history(save_path=plot_path)

        # Save training curves data for analysis
        curves_path = os.path.join(save_dir, 'training_curves.npy')
        try:
            np.save(curves_path, {
                'train_loss': [x['total'] for x in trainer.train_history],
                'val_auroc': [x['auroc'] for x in trainer.val_history],
                'val_accuracy': [x['accuracy'] for x in trainer.val_history],
                'val_rmse': [x['rmse'] for x in trainer.val_history],
                'routing_percent': [x.get('routing_to_gain', 0) for x in trainer.val_history]
            })
        except Exception as e:
            print(f"Error saving training curves: {e}")

        print(f"\n🎉 Experiment completed successfully!")
        print(f"Results saved to: {save_dir}")
        print(f"Total training time: {trainer.total_training_time:.1f} seconds")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user - saving current progress...")
        # Save interrupted model
        if 'model' in locals():
            interrupted_path = os.path.join(save_dir, 'interrupted_model.pth')
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': cfg,
                    'epoch': len(trainer.train_history) if hasattr(trainer, 'train_history') else 0,
                }, interrupted_path)
                print(f"Interrupted model saved to '{interrupted_path}'")
            except Exception as e:
                print(f"Error saving interrupted model: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()