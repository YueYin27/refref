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

import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal, Optional, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
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


@dataclass
class R3FPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: R3FPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = RefRefDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = R3FModelConfig()
    """specifies the model config"""


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
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                # if output_path is not None:
                #     for key in image_dict.keys():
                #         image = image_dict[key]  # [H, W, C] order
                #         vutils.save_image(
                #             image.permute(2, 0, 1).cpu(), output_path / f"{image_prefix}_{key}_{idx:04d}.png"
                #         )
                # save depth map (replace transform_train/val.json to transform_test.json to get res on train/val set)
                #
                new_max = 11.5  # the max depth value in the dataset
                new_min = 1.0  # the min depth value in the dataset
                old_max = 1  # the max depth value in the depth map
                old_min = 0  # the min depth value in the depth map

                # get mask
                mask = batch["mask"].to('cuda')

                # convert gt depth map back to world coordinate and then to distance map
                depth_gt = batch["depth"] / 255.0  # ranging from 0 to 1
                depth_gt_world = new_min + (new_max - new_min) * (old_max - depth_gt) / (old_max - old_min)  # convert depth to world coordinate
                depth_gt_world = depth_gt_world.to('cuda')
                directions_norm = camera_ray_bundle.metadata["directions_norm"].to('cuda')  # the norm of the directions
                distance_gt = depth_gt_world * directions_norm  # convert depth to distance map in world coordinate

                # convert predicted distance map to world coordinate
                distance_pred = images_dict["depth"].to(torch.float64)  # predicted distance map in world coordinate before scaling
                distance_pred /= self.scale_factor  # predicted distance map in world coordinate
                distance_pred *= directions_norm ** 2  # convert depth to distance map

                # # ONLY FOR ORACLE AND PYTHIA
                distance_pred = torch.where(mask == 1, distance_gt, distance_pred)

                metrics_dict["distance_l1"] = torch.nn.functional.l1_loss(distance_gt, distance_pred)

                masked_l1_loss = torch.nn.functional.l1_loss(distance_gt * mask, distance_pred * mask)
                metrics_dict["masked_distance_l1"] = masked_l1_loss

                # normalise distance maps between (0, 1)
                white, black = 1.0, 0.0
                distance_gt_normalised = black + (white - black) * (new_max - distance_gt) / (new_max - new_min)
                distance_pred_normalised = black + (white - black) * (new_max - distance_pred) / (new_max - new_min)

                depth_dir = os.path.join(output_path, 'distance_maps')
                if not os.path.exists(depth_dir):
                    os.makedirs(depth_dir)
                plt.imsave(os.path.join(depth_dir, 'r_' + str(idx) + '_depth_gt.png'),
                           distance_gt_normalised.cpu().squeeze().numpy(), cmap='gray')
                plt.imsave(os.path.join(depth_dir, 'r_' + str(idx) + '_depth_pred.png'),
                           distance_pred_normalised.cpu().squeeze().numpy(), cmap='gray')
                # save rgb images
                rgb_img = images_dict['img'].cpu().numpy()
                # clip the rgb image to 0-1 using
                rgb_img = np.clip(rgb_img, 0, 1)
                rgb_dir = os.path.join(output_path, 'rgb_images/')
                if not os.path.exists(rgb_dir):
                    os.makedirs(rgb_dir)
                plt.imsave(os.path.join(rgb_dir, 'r_' + str(idx) + '.png'), rgb_img)

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start))
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width))
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )

        self.train()
        return metrics_dict