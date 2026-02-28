import torch
import numpy as np
import os
import argparse
from PIL import Image
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize metric calculators
psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_fn = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate metrics for a scene")
    parser.add_argument("--scene_name", required=True, help="Scene name for reporting")
    parser.add_argument("--result_dir", required=True, help="Directory containing the scene's data (with rgb_images subdir)")
    parser.add_argument("--mask_dir", required=True, help="Directory containing mask images")
    parser.add_argument("--output_json", default="scene_metrics.json", help="Output JSON file path to save metrics")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to process")
    return parser.parse_args()


def load_image(path: str, mode: str = "RGB") -> torch.Tensor:
    img = Image.open(path).convert(mode)
    img = np.array(img)
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    return torch.from_numpy(img).float() / 255.0


def load_split_image(path: str) -> (torch.Tensor, torch.Tensor):
    img = load_image(path, mode="RGB")
    gt_rgb = img[:, :1512, :]
    pred_rgb = img[:, 1512:, :]
    return gt_rgb, pred_rgb


def load_mask(path: str) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = torch.from_numpy(np.array(mask)).float()
    mask = (mask > 127.5).float()
    return mask.unsqueeze(-1)


def compute_metrics(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, mask: torch.Tensor = None) -> dict:
    gt = gt_rgb.permute(2, 0, 1).unsqueeze(0).to(device)
    pred = pred_rgb.permute(2, 0, 1).unsqueeze(0).to(device)

    metrics = {
        'psnr': float(psnr_fn(gt, pred).item()),
        'ssim': float(ssim_fn(gt, pred).item()),
        'lpips': float(lpips_fn(gt, pred))
    }

    if mask is not None:
        mask_exp = mask.unsqueeze(0).permute(0, 3, 1, 2).to(device)
        mask_exp = mask_exp.expand_as(gt)

        masked_gt = gt * mask_exp
        masked_pred = pred * mask_exp

        # Masked PSNR
        if mask_exp.sum() > 0:
            mse = torch.sum((masked_gt - masked_pred)**2) / torch.sum(mask_exp)
            metrics['masked_psnr'] = float(10 * torch.log10(1.0 / mse))
        else:
            metrics['masked_psnr'] = float('nan')

        # Masked SSIM
        metrics['masked_ssim'] = float(ssim_fn(masked_gt, masked_pred).item())
        # Masked LPIPS
        metrics['masked_lpips'] = float(lpips_fn(masked_gt, masked_pred))

    return metrics


def aggregate_metrics(metrics_list, key):
    values = [m.get(key, np.nan) for m in metrics_list]
    valid_values = [v for v in values if not np.isnan(v)]
    return {
        'mean': float(np.mean(valid_values)) if valid_values else float('nan'),
        'count': len(valid_values)
    }


if __name__ == "__main__":
    args = parse_args()
    all_metrics = []

    for idx in [1,4,6,8,9]:
        rgb_path = os.path.join(args.result_dir, "rgb_images", f"r_{idx}.png")
        mask_path = os.path.join(args.mask_dir, f"frame_{idx}.png")

        try:
            gt_rgb, pred_rgb = load_split_image(rgb_path)
        except FileNotFoundError:
            print(f"RGB image {rgb_path} not found, skipping...")
            continue

        mask = load_mask(mask_path) if os.path.exists(mask_path) else None
        print(f"Loaded mask {mask_path}")

        assert pred_rgb.shape == gt_rgb.shape, f"RGB shape mismatch in {rgb_path}"
        if mask is not None:
            assert mask.shape[:2] == gt_rgb.shape[:2], f"Mask shape mismatch: {mask_path}"

        metrics = compute_metrics(pred_rgb, gt_rgb, mask)
        all_metrics.append(metrics)

        # # Print per-sample PSNR
        # print(f"Sample r_{idx}.png: PSNR = {metrics['psnr']:.4f}")

    results = {
        'Scene Name': args.scene_name,
        'PSNR': aggregate_metrics(all_metrics, 'psnr'),
        'SSIM': aggregate_metrics(all_metrics, 'ssim'),
        'LPIPS': aggregate_metrics(all_metrics, 'lpips'),
        'masked_PSNR': aggregate_metrics(all_metrics, 'masked_psnr'),
        'masked_SSIM': aggregate_metrics(all_metrics, 'masked_ssim'),
        'masked_LPIPS': aggregate_metrics(all_metrics, 'masked_lpips'),
    }

    print(f"\nResults for {args.scene_name}:")
    for metric, data in results.items():
        if metric != "Scene Name":
            print(f"{metric:<20} | Mean: {data['mean']:.4f} (based on {data['count']} samples)")

    with open(args.output_json, "w") as jf:
        json.dump(results, jf, indent=4)

    print(f"\nResults saved to {args.output_json}")
