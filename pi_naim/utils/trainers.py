# pi_naim/utils/trainers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import numpy as np
from tqdm import tqdm
import time
import json
import datetime
from typing import Dict, List
import os

from pi_naim.utils.masking import curriculum_mask
from .curriculum import Curriculum


class PINAIMTrainer:
    def __init__(self, model, optimizer, device="cpu", curriculum_cfg=None, lr_critic: float = 0.0001):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.curriculum = Curriculum(curriculum_cfg) if curriculum_cfg else None
        self.best_metrics = {}
        self.train_history = []
        self.val_history = []
        self.val_loader = None
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.total_training_time = 0.0

        # Adversarial training setup
        self.critic_optimizer = torch.optim.Adam(
            self.model.gain.critic.parameters(),
            lr=lr_critic,
            betas=(0.5, 0.9)
        )
        self.n_critic = getattr(model.cfg, 'n_critic', 5)

    def _ensure_directory_exists(self, file_path):
        """Ensure the directory for the given file path exists"""
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def set_val_loader(self, val_loader):
        """Set validation loader for routing statistics"""
        self.val_loader = val_loader

    def apply_curriculum_mask(self, X: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Apply curriculum masking based on the current epoch.
        """
        if self.curriculum:
            phase = self.curriculum.phase(epoch)
            return curriculum_mask(X, phase, device=self.device)
        else:
            # Default to MCAR if no curriculum is provided
            return curriculum_mask(X, "mcar", device=self.device)

    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        total_losses = {'total': 0.0, 'imp': 0.0, 'task': 0.0, 'adv_g': 0.0, 'adv_d': 0.0, 'reg': 0.0}
        num_batches = 0

        # Learning rate warmup for first 10 epochs
        if epoch < 10:
            warmup_factor = (epoch + 1) / 10
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * warmup_factor
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = self.critic_optimizer.param_groups[0]['lr'] * warmup_factor

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            X = batch["X_true"].to(self.device)
            y = batch["y"].to(self.device)
            mask_true = batch["mask"].to(self.device)

            # Curriculum masking
            mask_curriculum = self.apply_curriculum_mask(X, epoch)
            mask_combined = mask_true * mask_curriculum

            # Add periodic debugging
            if epoch % 10 == 0 and batch_idx == 0:  # First batch of every 10th epoch
                print(f"\n=== EPOCH {epoch} DEBUG ===")
                self.model.debug_loss_components(X, mask_combined, y, self.model(X, mask_combined))
                print("========================")

            # Train critic more frequently
            if batch_idx % self.n_critic == 0:
                self.critic_optimizer.zero_grad()
                loss_critic = self.model.train_adversarial(X, mask_combined)
                if loss_critic != 0:
                    loss_critic.backward()
                    # Enhanced gradient clipping for critic
                    torch.nn.utils.clip_grad_norm_(self.model.gain.critic.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_value_(self.model.gain.critic.parameters(), clip_value=0.5)
                    self.critic_optimizer.step()
                    total_losses['adv_d'] += loss_critic.item()

            # Train generator and main model
            self.optimizer.zero_grad()
            outputs = self.model(X, mask_combined)

            losses = self.model.compute_losses(X, mask_combined, y, outputs)

            # Check for invalid loss before backward
            if torch.isnan(losses['total']) or torch.isinf(losses['total']) or losses['total'] < 0:
                print(f"WARNING: Skipping batch due to invalid loss: {losses['total'].item()}")
                continue

            losses['total'].backward()

            # Enhanced gradient clipping with tighter values
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)

            self.optimizer.step()

            # Accumulate losses
            for k in losses:
                if k in total_losses:
                    total_losses[k] += losses[k].item()

            num_batches += 1

            # Update progress
            progress_bar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'task': f"{losses['task'].item():.4f}",
                'imp': f"{losses['imp'].item():.4f}"
            })

        # Average losses
        if num_batches > 0:
            return {k: v / num_batches for k, v in total_losses.items()}
        return total_losses

    def evaluate(self, dataloader, return_uncertainty=False):
        self.model.eval()
        results = {
            'y_true': [], 'y_pred': [], 'y_prob': [],
            'imp_errors': [], 'routing_decisions': [], 'missing_rates': [],
            'uncertainties': []
        }

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating', leave=False):
                X = batch["X_true"].to(self.device)
                y = batch["y"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(X, mask, return_uncertainty=return_uncertainty)

                # Store predictions
                results['y_true'].extend(y.cpu().numpy())
                results['y_pred'].extend(outputs['logits'].argmax(dim=1).cpu().numpy())
                results['y_prob'].extend(torch.softmax(outputs['logits'], dim=1)[:, 1].cpu().numpy())

                # Imputation error
                imp_error = F.mse_loss(outputs['x_imp'] * (1 - mask), X * (1 - mask))
                results['imp_errors'].append(imp_error.item())

                # Routing statistics
                results['routing_decisions'].extend(outputs['route_gain'].flatten().cpu().numpy())
                mr = 1.0 - (mask.sum(dim=1) / mask.shape[1])
                results['missing_rates'].extend(mr.cpu().numpy())

                if return_uncertainty:
                    results['uncertainties'].extend(outputs['uncertainty'].cpu().numpy())

        # Calculate metrics
        metrics = {
            'auroc': float(roc_auc_score(results['y_true'], results['y_prob'])),
            'accuracy': float(accuracy_score(results['y_true'], results['y_pred'])),
            'rmse': float(np.sqrt(np.mean(results['imp_errors']))),
            'routing_to_gain': float(np.mean(results['routing_decisions'])),
            'missing_rate': float(np.mean(results['missing_rates'])),
            'num_samples': len(results['y_true'])
        }

        if return_uncertainty:
            metrics['uncertainty'] = float(np.mean(results['uncertainties']))

        return metrics

    def train(self, train_loader, val_loader, epochs: int,
              scheduler=None, early_stopping_patience=10,
              checkpoint_path='best_model.pth'):
        best_auroc = 0.0
        patience_counter = 0
        history = {'train': [], 'val': []}

        # Set validation loader for routing statistics
        self.set_val_loader(val_loader)

        print(f"Starting training for {epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_losses = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start
            self.total_training_time += epoch_duration

            # Update history
            history['train'].append(train_losses)
            history['val'].append(val_metrics)
            self.train_history.append(train_losses)
            self.val_history.append(val_metrics)

            # Print progress
            print(f'\nEpoch {epoch + 1}/{epochs} ({epoch_duration:.1f}s):')
            print(f'  Train Loss: {train_losses["total"]:.4f} '
                  f'(Imp: {train_losses["imp"]:.4f}, Task: {train_losses["task"]:.4f}, Reg: {train_losses["reg"]:.4f})')
            print(f'  Val AUROC: {val_metrics["auroc"]:.4f}, '
                  f'Val Accuracy: {val_metrics["accuracy"]:.4f}, '
                  f'Val RMSE: {val_metrics["rmse"]:.4f}')

            # Print routing statistics if available
            if hasattr(self.model, 'route'):
                # Calculate average routing decisions
                route_gains = []
                with torch.no_grad():
                    for batch in val_loader:
                        X = batch["X_true"].to(self.device)
                        mask = batch["mask"].to(self.device)
                        route_result = self.model.route(mask)
                        # route returns a tuple (route_decisions, adaptive_threshold)
                        # We only want the route_decisions
                        if isinstance(route_result, tuple):
                            route_decisions = route_result[0]  # Get the first element (route_decisions)
                        else:
                            route_decisions = route_result
                        route_gains.extend(route_decisions.cpu().numpy())

                if route_gains:
                    avg_route_gain = np.mean(route_gains)
                    print(f'  Avg Route to GAIN: {avg_route_gain:.3f} '
                          f'({np.mean([x > 0.5 for x in route_gains]) * 100:.1f}% samples)')

            # Early stopping and model saving
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                patience_counter = 0
                self.best_metrics = val_metrics.copy()

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_losses,
                    'val_metrics': val_metrics,
                    'best_auroc': best_auroc
                }, checkpoint_path)
                print(f'  ✓ New best model saved (AUROC: {best_auroc:.4f})')
            else:
                patience_counter += 1
                print(f'  Patience counter: {patience_counter}/{early_stopping_patience}')

            # Check early stopping
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered at epoch {epoch + 1}')
                break

            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['auroc'])
                else:
                    scheduler.step()

                # Print learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'  Learning rate: {current_lr:.6f}')

        # Training completed
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time:.1f} seconds')

        # Load best model
        try:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")
        except Exception as e:
            print(f"Could not load best model: {e}")
            print("Using final model for evaluation")

        return history

    def get_training_summary(self):
        """Return a summary of the training process"""
        if not self.train_history or not self.val_history:
            return {}

        best_epoch = np.argmax([metrics['auroc'] for metrics in self.val_history])

        return {
            'best_epoch': int(best_epoch + 1),
            'best_auroc': float(self.val_history[best_epoch]['auroc']),
            'best_accuracy': float(self.val_history[best_epoch]['accuracy']),
            'final_train_loss': float(self.train_history[-1]['total']),
            'final_val_auroc': float(self.val_history[-1]['auroc']),
            'num_epochs': int(len(self.train_history))
        }

    def predict(self, dataloader, return_uncertainty=False):
        """Make predictions on new data"""
        self.model.eval()
        all_outputs = []

        with torch.no_grad():
            for batch in dataloader:
                X = batch["X_true"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(X, mask, return_uncertainty=return_uncertainty)
                all_outputs.append(outputs)

        return all_outputs

    def get_feature_importance(self, dataloader):
        """Calculate feature importance based on attention weights"""
        self.model.eval()
        feature_importances = []

        with torch.no_grad():
            for batch in dataloader:
                X = batch["X_true"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(X, mask)
                if 'attn_weights' in outputs:
                    attn = outputs['attn_weights'].cpu()

                    # Handle different attention formats
                    if attn.dim() == 3:  # (batch_size, seq_len, seq_len)
                        # Take mean across sequence and batch
                        attn_mean = attn.mean(dim=1).mean(dim=0)  # (seq_len,)
                        feature_importances.append(attn_mean.numpy())
                    elif attn.dim() == 2:  # (batch_size, features)
                        attn_mean = attn.mean(dim=0)  # (features,)
                        feature_importances.append(attn_mean.numpy())
                    else:
                        # Flatten and take mean
                        attn_flat = attn.view(attn.size(0), -1)
                        attn_mean = attn_flat.mean(dim=0)
                        feature_importances.append(attn_mean.numpy())

        if feature_importances:
            try:
                # Convert to numpy array and take mean across batches
                importance_array = np.array(feature_importances)
                overall_importance = importance_array.mean(axis=0)

                return overall_importance
            except Exception as e:
                print(f"Error calculating feature importance: {e}")
                return None
        else:
            return None

    def get_routing_statistics(self, dataloader):
        """Get detailed routing statistics"""
        self.model.eval()
        route_decisions = []
        missing_rates = []

        with torch.no_grad():
            for batch in dataloader:
                X = batch["X_true"].to(self.device)
                mask = batch["mask"].to(self.device)

                # Calculate missing rate
                mr = 1.0 - (mask.sum(dim=1) / mask.shape[1])
                missing_rates.extend(mr.cpu().numpy())

                # Get routing decisions
                if hasattr(self.model, 'route'):
                    route_result = self.model.route(mask)
                    # route returns a tuple (route_decisions, adaptive_threshold)
                    if isinstance(route_result, tuple):
                        route_gain = route_result[0]  # Get the first element
                    else:
                        route_gain = route_result
                    route_decisions.extend(route_gain.cpu().numpy())

        if route_decisions and missing_rates:
            stats = {
                'avg_route_to_gain': float(np.mean(route_decisions)),
                'percent_routed_to_gain': float(np.mean([x > 0.5 for x in route_decisions]) * 100),
                'avg_missing_rate': float(np.mean(missing_rates)),
                'std_missing_rate': float(np.std(missing_rates)),
                'routing_correlation': float(
                    np.corrcoef(missing_rates, route_decisions)[0, 1] if len(missing_rates) > 1 else 0)
            }
            return stats
        return None

    def print_performance_summary(self):
        """Print a beautiful performance summary"""
        summary = self.get_training_summary()
        routing_stats = self.get_routing_statistics(self.val_loader) if self.val_loader else None

        print("\n" + "=" * 60)
        print("           PI-NAIM PERFORMANCE SUMMARY")
        print("=" * 60)

        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Best AUROC:        {summary.get('best_auroc', 0):.4f}")
        print(f"   Best Accuracy:     {summary.get('best_accuracy', 0):.4f}")
        print(f"   Final Train Loss:  {summary.get('final_train_loss', 0):.4f}")

        # Calculate actual improvement over baseline (0.5 for random)
        improvement = (summary.get('best_auroc', 0) - 0.5) * 200
        print(f"   Improvement:       +{improvement:.2f}% over random baseline")

        print(f"\n⚙️  TRAINING DETAILS:")
        print(f"   Best Epoch:        {summary.get('best_epoch', 0)}")
        print(f"   Total Epochs:      {summary.get('num_epochs', 0)}")
        print(f"   Early Stopping:    {'Yes' if summary.get('num_epochs', 0) < 100 else 'No'}")
        print(f"   Training Time:     {self.total_training_time / 3600:.2f} hours")

        if routing_stats:
            print(f"\n🔄 ROUTING STATISTICS:")
            print(f"   Avg Route to GAIN: {routing_stats.get('avg_route_to_gain', 0):.3f}")
            print(f"   % to GAIN:         {routing_stats.get('percent_routed_to_gain', 0):.1f}%")
            print(f"   Avg Missing Rate:  {routing_stats.get('avg_missing_rate', 0):.3f}")
            print(f"   Routing Correlation: {routing_stats.get('routing_correlation', 0):.3f}")

        print(f"\n🤖 MODEL ARCHITECTURE:")
        print(f"   Parameters:        {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Device:            {str(self.device)}")

        print("\n" + "=" * 60)
        print("✅ PI-NAIM training completed successfully!")
        print("=" * 60)

    def save_training_report(self, file_path):
        """Save comprehensive training report"""
        try:
            # Convert numpy values to Python native types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj

            report = {
                'timestamp': datetime.datetime.now().isoformat(),
                'training_summary': convert_to_serializable(self.get_training_summary()),
                'best_metrics': convert_to_serializable(self.best_metrics),
                'model_parameters': int(sum(p.numel() for p in self.model.parameters())),
                'device': str(self.device),
                'training_time_hours': float(self.total_training_time / 3600),
                'final_val_metrics': convert_to_serializable(self.val_history[-1] if self.val_history else {})
            }

            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)

            print(f"Training report saved to {file_path}")

        except Exception as e:
            print(f"Error saving training report: {e}")

    def plot_training_history(self, save_path=None):
        """Plot training history (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt

            epochs = range(1, len(self.train_history) + 1)

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            # Plot losses
            train_losses = [x['total'] for x in self.train_history]
            ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # Plot AUROC
            val_aurocs = [x['auroc'] for x in self.val_history]
            ax2.plot(epochs, val_aurocs, 'r-', label='Validation AUROC')
            ax2.set_title('Validation AUROC')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('AUROC')
            ax2.legend()
            ax2.grid(True)

            # Plot accuracy
            val_accuracies = [x['accuracy'] for x in self.val_history]
            ax3.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
            ax3.set_title('Validation Accuracy')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy')
            ax3.legend()
            ax3.grid(True)

            # Plot component losses
            imp_losses = [x['imp'] for x in self.train_history]
            task_losses = [x['task'] for x in self.train_history]
            reg_losses = [x['reg'] for x in self.train_history]

            ax4.plot(epochs, imp_losses, 'orange', label='Imputation Loss')
            ax4.plot(epochs, task_losses, 'purple', label='Task Loss')
            ax4.plot(epochs, reg_losses, 'brown', label='Reg Loss')
            ax4.set_title('Component Losses')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.legend()
            ax4.grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                print(f"Training history plot saved to {save_path}")

            plt.show()

        except ImportError:
            print("Matplotlib not installed. Skipping plot generation.")
        except Exception as e:
            print(f"Error generating plot: {e}")