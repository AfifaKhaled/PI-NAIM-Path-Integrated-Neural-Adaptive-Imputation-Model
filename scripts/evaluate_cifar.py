import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import argparse
import os
import sys
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datasets.cifar_missing import get_cifar_missing_loaders
from models.models_cifar import CIFARImputationModel, CIFARClassifier



def evaluate_cifar(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    _, test_loader = get_cifar_missing_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        missing_type=args.missing_type,
        missing_rate=args.missing_rate
    )

    # Load model
    model = CIFARImputationModel().to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Metrics
    total_psnr = 0
    total_ssim = 0
    total_mse = 0
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            incomplete = batch['incomplete'].to(device)
            mask = batch['mask'].to(device)
            complete = batch['complete'].to(device)

            imputed, _ = model(incomplete, mask)

            # Calculate metrics only on missing regions
            for i in range(imputed.size(0)):
                img_comp = complete[i].cpu().numpy().transpose(1, 2, 0)
                img_imp = imputed[i].cpu().numpy().transpose(1, 2, 0)
                img_mask = mask[i].cpu().numpy().transpose(1, 2, 0)

                # Only evaluate on missing pixels
                missing_indices = img_mask[..., 0] == 0  # All channels have same mask

                if np.sum(missing_indices) > 0:
                    # PSNR and SSIM on complete image for comparison
                    total_psnr += psnr(img_comp, img_imp, data_range=1.0)

                    # SSIM per channel and average
                    ssim_val = 0
                    for ch in range(3):
                        ssim_val += ssim(img_comp[..., ch], img_imp[..., ch], data_range=1.0)
                    total_ssim += ssim_val / 3

                    # MSE on missing regions
                    mse = np.mean((img_comp[missing_indices] - img_imp[missing_indices]) ** 2)
                    total_mse += mse

                    count += 1

    print(f"Evaluation Results:")
    print(f"PSNR: {total_psnr / count:.4f}")
    print(f"SSIM: {total_ssim / count:.4f}")
    print(f"MSE on missing regions: {total_mse / count:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--missing_type', type=str, default='random')
    parser.add_argument('--missing_rate', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    evaluate_cifar(args)