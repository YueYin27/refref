import torch
import numpy as np
import os
import argparse
from PIL import Image
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import csv
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize metric calculators
psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate metrics for a scene")
    parser.add_argument("--scene_name", required=True, help="Scene name for reporting")
    parser.add_argument("--result_dir", required=True, help="Directory containing the scene's data (with rgb_images and distance_maps subdirs)")
    parser.add_argument("--mask_dir", required=True, help="Directory containing mask images")
    parser.add_argument("--output_csv", default="scene_metrics.csv", help="Output CSV file path to save metrics")
    return parser.parse_args()


def compute_metrics(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, mask: torch.Tensor = None) -> dict:
    gt = gt_rgb.permute(2, 0, 1).unsqueeze(0).to(device)
    pred = pred_rgb.permute(2, 0, 1).unsqueeze(0).to(device)

    psnr = psnr_fn(gt, pred)
    ssim = ssim_fn(gt, pred)
    lpips = lpips_fn(gt, pred)

    metrics = {
        'psnr': float(psnr.item()),
        'ssim': float(ssim.item()),
        'lpips': float(lpips)
    }

    if mask is not None:
        mask = mask.unsqueeze(0).permute(0, 3, 1, 2).to(device)
        mask = mask.expand_as(gt)

        masked_gt = gt[mask == 1]
        masked_pred = pred[mask == 1]

        if len(masked_gt) > 0:
            mse = torch.mean((masked_gt - masked_pred) ** 2)
            masked_psnr = 10 * torch.log10(1.0 / mse)
            metrics['masked_psnr'] = float(masked_psnr)
        else:
            metrics['masked_psnr'] = float('nan')

    return metrics


def compute_distance_metrics(pred_dist: torch.Tensor, gt_dist: torch.Tensor, mask: torch.Tensor = None) -> dict:
    """Compute all distance-related metrics (MSE and MAE, both regular and masked)"""
    metrics = {}

    # Convert to device if not already
    pred_dist = pred_dist.to(device)
    gt_dist = gt_dist.to(device)

    # Convert distance maps back to world coordinates
    gt_dist = 11.5 - 10.5 * gt_dist
    pred_dist = 11.5 - 10.5 * pred_dist

    # Regular distance metrics
    # metrics['dist_mse'] = float(torch.mean((pred_dist - gt_dist) ** 2).item())
    metrics['dist_rmse'] = float(torch.sqrt(torch.mean((pred_dist - gt_dist) ** 2)).item())
    metrics['dist_mae'] = torch.nn.functional.l1_loss(pred_dist, gt_dist).item()

    # Masked distance metrics if mask is provided
    # if mask is not None:
        # Prepare mask (single channel, binary)
    mask = mask.squeeze(-1)  # Remove channel dim if present
    mask = (mask > 0.5).float().to(device)

    # Reshape tensors for masking
    pred_dist = pred_dist.unsqueeze(0).unsqueeze(0)
    gt_dist = gt_dist.unsqueeze(0).unsqueeze(0)
    mask = mask.unsqueeze(0).unsqueeze(0)

    # Apply mask
    masked_pred = pred_dist[mask == 1]
    masked_gt = gt_dist[mask == 1]

    # if len(masked_gt) > 0:
    metrics['masked_dist_rmse'] = float(torch.sqrt(torch.mean((masked_pred - masked_gt) ** 2)).item())
    metrics['masked_dist_mae'] = float(torch.mean(torch.abs(masked_pred - masked_gt)).item())

    return metrics


def load_image(path: str, mode: str = "RGB") -> torch.Tensor:
    img = Image.open(path).convert(mode)
    img = np.array(img)
    if len(img.shape) == 2:
        img = img[..., np.newaxis]  # Add channel dimension if grayscale
    return torch.from_numpy(img).float() / 255.0


def load_split_image(path: str) -> (torch.Tensor, torch.Tensor):
    img = load_image(path, mode="RGB")
    # Split into left (GT) and right (pred)
    gt_rgb = img[:, :800, :]
    pred_rgb = img[:, 800:, :]
    return gt_rgb, pred_rgb


def load_split_distance_image(path: str) -> (torch.Tensor, torch.Tensor):
    """Load a combined distance map image (same layout as RGB: left=GT, right=pred) and split it."""
    img = load_image(path, mode="L")
    # Split into left (GT) and right (pred), same 800-pixel split as load_split_image
    gt_dist = img[:, :800, :]
    pred_dist = img[:, 800:, :]
    return gt_dist, pred_dist


def load_mask(path: str) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = torch.from_numpy(np.array(mask)).float()
    mask = (mask > 127.5).float()
    return mask.unsqueeze(-1)  # Add channel dimension


if __name__ == "__main__":
    args = parse_args()

    all_metrics = []
    for idx in range(100):
        # Generate filenames
        rgb_path = os.path.join(args.result_dir, "rgb_images", f"r_{idx}.png")
        mask_path = os.path.join(args.mask_dir, f"r_{idx}_mask_0000.png")
        dist_path = os.path.join(args.result_dir, "distance_maps", f"r_{idx}_dist.png")

        # Load data
        try:
            gt_rgb, pred_rgb = load_split_image(rgb_path)
        except FileNotFoundError:
            print(f"RGB image {rgb_path} not found, skipping...")
            continue

        mask = load_mask(mask_path) if os.path.exists(mask_path) else None

        # Initialize distance metrics as NaN
        dist_metrics = {
            'dist_mse': float('nan'),
            'dist_mae': float('nan'),
            'masked_dist_mse': float('nan'),
            'masked_dist_mae': float('nan')
        }

        try:
            gt_dist, pred_dist = load_split_distance_image(dist_path)
            dist_metrics = compute_distance_metrics(pred_dist, gt_dist, mask)
        except FileNotFoundError as e:
            print(f"Distance map file not found: {e}, skipping distance metrics for index {idx}")

        # Check shape consistency
        assert pred_rgb.shape == gt_rgb.shape, f"RGB shape mismatch in {rgb_path}"
        if 'dist_rmse' in dist_metrics and not np.isnan(dist_metrics['dist_rmse']):
            assert pred_dist.shape == gt_dist.shape, f"Distance map shape mismatch in {dist_path}"
        if mask is not None:
            assert mask.shape[:2] == gt_rgb.shape[:2], f"Mask shape mismatch: {mask_path}"

        # Compute RGB metrics
        metrics = compute_metrics(pred_rgb, gt_rgb, mask)

        # Combine all metrics
        metrics.update(dist_metrics)
        all_metrics.append(metrics)


    def aggregate_metrics(metrics_list, key):
        values = [m.get(key, np.nan) for m in metrics_list]
        valid_values = [v for v in values if not np.isnan(v)]
        return {
            'mean': np.mean(valid_values) if valid_values else float('nan'),
            'count': len(valid_values)
        }


    results = {
        'PSNR': aggregate_metrics(all_metrics, 'psnr'),
        'SSIM': aggregate_metrics(all_metrics, 'ssim'),
        'LPIPS': aggregate_metrics(all_metrics, 'lpips'),
        'masked_PSNR': aggregate_metrics(all_metrics, 'masked_psnr'),
        'Distance_RMSE': aggregate_metrics(all_metrics, 'dist_rmse'),
        'Distance_MAE': aggregate_metrics(all_metrics, 'dist_mae'),
        'Masked_Distance_RMSE': aggregate_metrics(all_metrics, 'masked_dist_rmse'),
        'Masked_Distance_MAE': aggregate_metrics(all_metrics, 'masked_dist_mae')
    }

    print(f"\nResults for {args.scene_name}:")
    for metric, data in results.items():
        print(f"{metric:<20} | Mean: {data['mean']:.4f} (based on {data['count']} samples)")

    file_exists = os.path.isfile(args.output_csv)

    with open(args.output_csv, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Scene Name", "PSNR", "SSIM", "LPIPS",
                "Masked PSNR", "DRMSE", "DMAE",
                "Masked DRMSE", "Masked DMAE"
            ])

        writer.writerow([
            args.scene_name,
            results['PSNR']['mean'],
            results['SSIM']['mean'],
            results['LPIPS']['mean'],
            results['masked_PSNR']['mean'],
            results['Distance_RMSE']['mean'],
            results['Distance_MAE']['mean'],
            results['Masked_Distance_RMSE']['mean'],
            results['Masked_Distance_MAE']['mean']
        ])

    print(f"\nResults saved to {args.output_csv}")

    # convert csv to xlsx
    df = pd.read_csv(args.output_csv)
    xlsx_path = args.output_csv.replace('.csv', '.xlsx')
    df.to_excel(xlsx_path, index=False)
    print(f"Results saved to {xlsx_path}")