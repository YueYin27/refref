# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications made in 2025 by Yue Yin and Enze Tao (The Australian National University).
# These changes are part of the research presented in the paper:
# RefRef: A Synthetic Dataset and Benchmark for Reconstructing Refractive and Reflective Objects (https://arxiv.org/abs/2505.05848)


"""
Nerfstudio R3F Pipeline
"""
from __future__ import annotations

import json
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal, Optional, Type

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, PngImagePlugin
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from r3f_ns.refref_datamanager import RefRefDataManagerConfig
from r3f_ns.r3f_model import R3FModel, R3FModelConfig


def _predicted_distance(outputs, batch, directions_norm, scale_factor, device):
    """Predicted distance along the ray, in pre-scale world units.

    Same construction as the GT branch below, minus the GT-depth fill values: FG stage
    composites the ray-traced mesh surface over the BG field, everything else converts the
    model's camera-z depth with directions_norm. The object mask picks the FG pixels when
    one is available (the real captures have SAM3 masks but no GT depth, so they land
    here); without a mask, every finite surface hit is taken.
    """
    def _squeeze(name):
        t = outputs[name].to(device=device, dtype=torch.float64)
        return t.squeeze(-1) if t.ndim == 3 and t.shape[-1] == 1 else t

    if "fg_surface_depth" in outputs and "bg_depth" in outputs:
        # camera-z -> ray distance for the BG field; t_entry is already a ray distance.
        distance = _squeeze("bg_depth") / scale_factor * directions_norm
        fg_surface = _squeeze("fg_surface_depth") / scale_factor
        fg_valid = torch.isfinite(fg_surface)
        if "mask" in batch:
            # Squeezed here rather than through the caller's _as_hw_tensor, which is nested
            # inside get_average_image_metrics and not in scope. A mask that does not match
            # the render is ignored instead of raising: every finite mesh hit is then taken,
            # which is what happens for a scene with no mask at all.
            mask = batch["mask"]
            mask = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
            mask = mask.to(device=device).squeeze()
            if mask.shape == fg_valid.shape:
                fg_valid = fg_valid & (mask > 0.5)
        distance = distance.clone()
        distance[fg_valid] = fg_surface[fg_valid]
    else:
        distance = _squeeze("depth") / scale_factor * directions_norm
    return distance


def _save_jpg(path: str, image: np.ndarray) -> None:
    """Write a float [0,1] image as JPEG, creating the directory. Mirrors
    scripts/eval_render.py so R3F's real-capture renders match the baselines'."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = (np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr).convert("RGB").save(path, quality=95, subsampling=0)


def _save_distance_jpg(path: str, distance: torch.Tensor):
    """Normalise a distance map per image (near white, far black) and write it as JPEG.

    Returns the {min, max} it used so distance_maps/ranges.json can invert the mapping;
    the real captures have no GT distance, so there is no fixed range to map onto.
    """
    dist = distance.detach().cpu().numpy()
    finite = np.isfinite(dist)
    if not finite.any():
        return None
    # Robust range, not min/max: rays that hit nothing run to the far plane (~10^5 world
    # units once un-scaled) and would push everything real onto white. Three times the
    # median's spread above it, capped at p99. Mirrors scripts/eval_render.robust_range.
    p1, p25, p50, p99 = np.percentile(dist[finite], [1, 25, 50, 99])
    lo = float(p1)
    hi = min(float(p99), float(p50 + 3.0 * (p50 - p25)))
    if not hi > lo:
        hi = lo + 1.0
    span = hi - lo
    norm = np.where(finite, 1.0 - (np.clip(dist, lo, hi) - lo) / span, 0.0)
    _save_jpg(path, norm)
    return {"min": lo, "max": hi}


def _save_png_no_metadata(path: str, image: np.ndarray) -> None:
    """Save PNG without metadata to avoid libpng duplicate eXIf warnings."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if arr.size > 0 and arr.max() <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    pnginfo = PngImagePlugin.PngInfo()
    Image.fromarray(arr).save(path, pnginfo=pnginfo)


@dataclass
class R3FPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: R3FPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = RefRefDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = R3FModelConfig()
    """specifies the model config"""
    stage: Literal["bg", "fg", "none"] = "none"
    """Training stage: 'bg' trains background only (masking out foreground object), 'fg' trains foreground in-object field."""
    bg_checkpoint_path: Optional[str] = None
    """Path to background field checkpoint for fg stage (e.g., .../step-000010000.ckpt)."""
    bg_far: Optional[float] = None
    """Far plane used when training the BG model. Required for unbounded scenes
    where BG was trained with a larger far than FG (e.g. BG far=1000, FG far=15)."""
    bg_opaque_background: bool = True
    """Whether frozen BG field should use opaque background when queried in fg stage.
    Should match BG training config (refref_hdr.gin uses True)."""


class R3FPipeline(VanillaPipeline):
    """R3F Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: R3FPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        # Propagate stage and bg checkpoint to datamanager and model configs
        config.datamanager.stage = config.stage
        config.model.stage = config.stage
        if config.bg_checkpoint_path is not None:
            config.model.bg_checkpoint_path = config.bg_checkpoint_path
        if config.bg_far is not None:
            config.model.bg_far = config.bg_far
        config.model.bg_opaque_background = config.bg_opaque_background

        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            ply_path=self.datamanager.dataparser.ply_path,
            scale_factor=self.datamanager.dataparser.scale_factor,
        )
        self.model.to(device)

        # Store for later use
        self.ply_path = self.datamanager.dataparser.ply_path
        self.scale_factor = self.datamanager.dataparser.scale_factor

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                R3FModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

    @profiler.time_function
    def get_average_image_metrics(
            self,
            data_loader,
            image_prefix: str,
            step: Optional[int] = None,
            output_path: Optional[Path] = None,
            get_std: bool = False,
    ):
        """Iterate over all the images in the dataset and get the average.

        Args:
            data_loader: the data loader to iterate over
            image_prefix: prefix to use for the saved image filenames
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        distance_ranges = {}
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            idx = 0
            # for camera, batch in data_loader:
            for camera_ray_bundle, batch in data_loader:
                inner_start = time()
                # Some dataloaders return a `Cameras` object instead of a RayBundle.
                # The model expects a RayBundle with `.directions`; convert if necessary.
                if not hasattr(camera_ray_bundle, "directions"):
                    if hasattr(camera_ray_bundle, "generate_rays"):
                        try:
                            camera_ray_bundle = camera_ray_bundle.generate_rays(camera_indices=0, keep_shape=True)
                        except TypeError:
                            # fallback: some implementations expect an image index instead
                            camera_ray_bundle = camera_ray_bundle.generate_rays(keep_shape=True)
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)

                new_max = 11.5  # the max depth value in the dataset
                new_min = 1.0  # the min depth value in the dataset
                old_max = 1  # the max depth value in the depth map
                old_min = 0  # the min depth value in the depth map

                # ensure tensors are on the same device/dtype as the model
                device = next(self.model.parameters()).device

                def _as_hw_tensor(value, name: str) -> torch.Tensor:
                    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                    tensor = tensor.to(device=device, dtype=torch.float64).squeeze()

                    if tensor.ndim == 1 and tensor.numel() == num_rays:
                        return tensor.reshape(height, width)

                    if tensor.ndim == 2:
                        if tensor.shape == (height, width):
                            return tensor
                        if tensor.shape == (width, height):
                            return tensor.transpose(0, 1)
                        if tensor.shape[0] == num_rays and tensor.shape[1] == 1:
                            return tensor[:, 0].reshape(height, width)

                    if tensor.ndim == 3:
                        if tensor.shape[0] == height and tensor.shape[1] == width:
                            return tensor[..., 0]
                        if tensor.shape[1] == height and tensor.shape[2] == width:
                            return tensor[0]

                    raise RuntimeError(
                        f"Unsupported shape for {name}: {tuple(tensor.shape)}. "
                        f"Expected a per-pixel map compatible with ({height}, {width})."
                    )

                has_gt_mask_depth = "mask" in batch and "depth" in batch
                if not has_gt_mask_depth:
                    # No GT depth: the real captures, and the synthetic vis split. Skip the
                    # distance METRICS (there is nothing to compare against) but still write
                    # the render and its distance map, in the layout every other method uses
                    # -- see scripts/eval_render.py, which does the same for the baselines:
                    #   <output_path>/rgb_images/r_<idx>.jpg      prediction only
                    #   <output_path>/distance_maps/r_<idx>.jpg   near = white, far = black
                    # jpg, because 100 frames x 60 scenes x 7 methods is a lot of PNG.
                    directions_norm = _as_hw_tensor(
                        camera_ray_bundle.metadata["directions_norm"], "directions_norm"
                    )
                    distance = _predicted_distance(outputs, batch, directions_norm,
                                                   self.scale_factor, device)
                    _save_jpg(
                        os.path.join(output_path, 'rgb_images', 'r_' + str(idx) + '.jpg'),
                        np.clip(outputs["rgb"].detach().cpu().numpy(), 0, 1),
                    )
                    rng = _save_distance_jpg(
                        os.path.join(output_path, 'distance_maps', 'r_' + str(idx) + '.jpg'),
                        distance,
                    )
                    if rng is not None:
                        distance_ranges[str(idx)] = rng

                    assert "num_rays_per_sec" not in metrics_dict
                    metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start))
                    fps_str = "fps"
                    assert fps_str not in metrics_dict
                    metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width))
                    metrics_dict_list.append(metrics_dict)
                    progress.advance(task)
                    idx = idx + 1
                    continue

                # get mask
                mask = _as_hw_tensor(batch["mask"], "mask")

                # convert gt depth map back to world coordinate and then to distance map
                depth_gt = _as_hw_tensor(batch["depth"], "depth") / 255.0  # ranging from 0 to 1
                depth_gt_world = new_min + (new_max - new_min) * (old_max - depth_gt) / (old_max - old_min)

                # get directions_norm and move to same device/dtype
                directions_norm = _as_hw_tensor(camera_ray_bundle.metadata["directions_norm"], "directions_norm")

                distance_gt = depth_gt_world * directions_norm

                # Build predicted distance map.
                if "fg_surface_depth" in outputs and "bg_depth" in outputs:
                    # FG stage: use exact mesh surface depth for FG pixels,
                    # BG field depth (camera-z * directions_norm) for everything else.
                    bg_dist = outputs["bg_depth"].to(device=device, dtype=torch.float64)
                    if bg_dist.ndim == 3 and bg_dist.shape[-1] == 1:
                        bg_dist = bg_dist.squeeze(-1)
                    bg_dist = bg_dist / self.scale_factor * directions_norm  # camera-z → ray distance
                    bg_dist = torch.nan_to_num(bg_dist, nan=new_max, posinf=new_max, neginf=new_min)

                    fg_surface = outputs["fg_surface_depth"].to(device=device, dtype=torch.float64)
                    if fg_surface.ndim == 3 and fg_surface.shape[-1] == 1:
                        fg_surface = fg_surface.squeeze(-1)
                    # t_entry was ray-cast with unit directions → already ray distance (in scaled coords)
                    fg_surface = fg_surface / self.scale_factor

                    # Composite: start from BG, overlay valid FG surface pixels using GT mask
                    distance_pred = bg_dist.clone()
                    fg_valid = (mask > 0.5) & torch.isfinite(fg_surface)
                    distance_pred[fg_valid] = fg_surface[fg_valid]
                else:
                    # BG-only or normal stage: model depth is camera-z, convert to ray distance
                    distance_pred = outputs["depth"].to(device=device, dtype=torch.float64)
                    if distance_pred.ndim == 3 and distance_pred.shape[-1] == 1:
                        distance_pred = distance_pred.squeeze(-1)
                    distance_pred = distance_pred / self.scale_factor * directions_norm
                    distance_pred = torch.nan_to_num(distance_pred, nan=new_max, posinf=new_max, neginf=new_min)

                # TODO: add GT mask for oracle and stage-1 mask for R3F
                # distance_pred = torch.where(mask == 1, distance_gt, distance_pred)

                metrics_dict["distance_l1"] = torch.nn.functional.l1_loss(distance_gt, distance_pred)

                masked_l1_loss = torch.nn.functional.l1_loss(distance_gt * mask, distance_pred * mask)
                metrics_dict["masked_distance_l1"] = masked_l1_loss

                # normalise distance maps between (0, 1), fixed mapping: 1.0->white, 11.5->black
                white, black = 1.0, 0.0
                distance_gt_vis = torch.clamp(distance_gt, min=new_min, max=new_max)
                distance_pred_vis = torch.clamp(distance_pred, min=new_min, max=new_max)
                distance_gt_normalised = black + (white - black) * (new_max - distance_gt_vis) / (new_max - new_min)
                distance_pred_normalised = black + (white - black) * (new_max - distance_pred_vis) / (new_max - new_min)
                distance_gt_normalised = torch.clamp(distance_gt_normalised, 0.0, 1.0)
                distance_pred_normalised = torch.clamp(distance_pred_normalised, 0.0, 1.0)

                depth_dir = os.path.join(output_path, 'distance_maps')
                if not os.path.exists(depth_dir):
                    os.makedirs(depth_dir)
                distance_pair = torch.cat([distance_gt_normalised, distance_pred_normalised], dim=1)
                distance_pair_u8 = (distance_pair.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
                _save_png_no_metadata(
                    os.path.join(depth_dir, 'r_' + str(idx) + '_depth.png'),
                    distance_pair_u8,
                )
                # save rgb images
                rgb_img = images_dict['img'].cpu().numpy()
                # clip the rgb image to 0-1 using
                rgb_img = np.clip(rgb_img, 0, 1)
                rgb_dir = os.path.join(output_path, 'rgb_images/')
                if not os.path.exists(rgb_dir):
                    os.makedirs(rgb_dir)
                _save_png_no_metadata(os.path.join(rgb_dir, 'r_' + str(idx) + '.png'), rgb_img)

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start))
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width))
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1

        if output_path is not None and distance_ranges:
            # Per-image normalisation is only invertible with the range it used.
            os.makedirs(os.path.join(output_path, 'distance_maps'), exist_ok=True)
            with open(os.path.join(output_path, 'distance_maps', 'ranges.json'), 'w') as f:
                json.dump({"fixed": False, "frames": distance_ranges}, f, indent=2)

        metrics_dict = {}
        for key in {key for frame in metrics_dict_list for key in frame}:
            values = torch.tensor(
                [frame[key] for frame in metrics_dict_list if key in frame],
                dtype=torch.float64,
            )
            values = values[torch.isfinite(values)]
            if values.numel() == 0:
                continue
            if get_std and values.numel() > 1:
                key_std, key_mean = torch.std_mean(values)
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(values.mean())

        self.train()
        return metrics_dict