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


import importlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type
from typing import Literal

import gin
import numpy as np
import open3d as o3d
import torch
import trimesh
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model
from nerfstudio.utils import colormaps
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from internal import train_utils, image
from internal.configs import Config
from internal.models import Model as r3f
from internal.ray_reflection import RayReflection


@dataclass
class R3FModelConfig(ModelConfig):
    gin_file: list = None
    """Config files list to load default setting of Model/NerfMLP/PropMLP"""
    compute_extras: bool = True
    """if True, compute extra quantities besides color."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    rand: bool = True
    """random number generator (or None for deterministic output)."""
    zero_glo: bool = False
    """if True, when using GLO pass in vector of zeros."""
    background_color: Literal["random", "black", "white"] = "white"
    """Whether to randomize the background color."""
    _target: Type = field(default_factory=lambda: R3FModel)

class R3FModel(Model):
    config: R3FModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        # update default setting
        # gin.parse_config_files_and_bindings(self.config.gin_file, None)
        gin_files = []
        for g in self.config.gin_file:
            if os.path.exists(g):
                gin_files.append(g)
            else:
                package_path = importlib.util.find_spec("r3f_ns").origin.split('/')[:-2]
                package_path = '/'.join(package_path)
                gin_files.append(package_path+'/'+g)
        gin.parse_config_files_and_bindings(gin_files, None)
        config = Config()

        # load mesh files and add to scene one by one
        ior_map = {"glass": 1.5, "water": 1.333, "diamond": 2.418, "air": 1.0, "alcohol": 1.36, "plastic": 1.45, "perfume": 1.46}
        scene = o3d.t.geometry.RaycastingScene()  # initialize a scene
        mesh_files = str(self.kwargs['ply_path']).split()
        self.scene = None
        if mesh_files:
            material_list = [
                next((word for word in path.split('_') if word in ior_map), path.split('_')[-1].replace('.ply', ''))
                for path in mesh_files
            ]
            iors = [ior_map[material] for material in material_list]  # Map materials to IOR values
            iors.append(float('nan'))  # Append NaN to the list
            iors = torch.tensor(iors, dtype=torch.float32)  # Convert to PyTorch tensor
            mesh_indices = []  # to store the idx of added meshes
            for idx, ply_path in enumerate(mesh_files):
                # Load and preprocess mesh
                mesh = trimesh.load_mesh(ply_path)
                vertices_tensor = o3d.core.Tensor(mesh.vertices * self.kwargs['scale_factor'], dtype=o3d.core.Dtype.Float32)
                triangles_tensor = o3d.core.Tensor(mesh.faces, dtype=o3d.core.Dtype.UInt32)
                mesh_index = scene.add_triangles(vertices_tensor, triangles_tensor)  # add mesh to scene and get index
                mesh_indices.append(mesh_index)
            self.scene = scene
            self.iors = iors
            self.mesh_indices = mesh_indices
            self.scene = {"scene": scene, "iors": iors, "mesh_indices": mesh_indices}

        self.r3f = r3f(config=config)

        self.collider = NearFarCollider(near_plane=self.r3f.config.near, far_plane=self.r3f.config.far)
        self.step = 0

        # Renderer
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def construct_batch_from_raybundle(self, ray_bundle):
        batch = {}
        batch['origins'] = ray_bundle.origins
        batch['directions'] = ray_bundle.directions * ray_bundle.metadata["directions_norm"]
        batch['viewdirs'] = ray_bundle.directions
        batch['radii'] = torch.sqrt(ray_bundle.pixel_area)* 2 / torch.sqrt(torch.full_like(ray_bundle.pixel_area,12.))
        batch['cam_idx'] = ray_bundle.camera_indices
        batch['near'] = ray_bundle.nears
        batch['far'] = ray_bundle.fars
        batch['cam_dirs'] = None  # did not be calculated in raybundle
        # batch['imageplane'] = None
        # batch['exposure_values'] = None
        return batch

    def get_outputs(self, ray_bundle: RayBundle):
        ray_bundle.metadata["viewdirs"] = ray_bundle.directions
        ray_bundle.metadata["radii"] = torch.sqrt(ray_bundle.pixel_area)* 2 / torch.sqrt(torch.full_like(ray_bundle.pixel_area,12.))
        ray_bundle.directions = ray_bundle.directions * ray_bundle.metadata["directions_norm"]

        if self.training:
            anneal_frac = np.clip(self.step / self.config.proposal_weights_anneal_max_num_iters, 0, 1)
        else:
            anneal_frac = 1.0
        batch = self.construct_batch_from_raybundle(ray_bundle)

        renderings, ray_history, rfls, ray_results, ray_samples = self.r3f(
                rand=self.config.rand if self.training else False,  # set to false when evaluating or rendering
                batch=batch,
                train_frac=anneal_frac,
                compute_extras=self.config.compute_extras,
                zero_glo=self.config.zero_glo if self.training else True, # set to True when evaluating or rendering
                reflection=None,
                scene=self.scene,)

        renderings_rfl, _, _, _, _ = self.r3f(
            rand=self.config.rand if self.training else False,  # set to false when evaluating or rendering
            batch=batch,
            train_frac=anneal_frac,
            compute_extras=self.config.compute_extras,
            zero_glo=self.config.zero_glo if self.training else True,  # set to True when evaluating or rendering
            reflection=rfls,)  # set to True when rendering reflection, False when rendering refraction

        if not self.training:
            renderings_depth = renderings
        else:
            renderings_straight, _, _, _, _ = self.r3f(
                rand=self.config.rand if self.training else False,  # set to false when evaluating or rendering
                batch=batch,
                train_frac=anneal_frac,
                compute_extras=self.config.compute_extras,
                zero_glo=self.config.zero_glo if self.training else True,  # set to True when evaluating or rendering
                straight=True,)
            renderings_depth = renderings_straight

        # Fresnel equation
        ray_samples_rfl = rfls[-1]
        normals = ray_samples_rfl.normals
        ray_reflection = RayReflection(ray_samples_rfl.origins, ray_samples_rfl.directions, ray_samples_rfl.get_positions(), 1.0 / 1.5)  # TODO: update n1/n2
        R = ray_reflection.fresnel_fn(normals)  # [8192, 1]
        rgb_rfl, rgb_rfr = renderings_rfl[2]['rgb'], renderings[2]['rgb']
        comp_rgb = R * rgb_rfl + (1 - R) * rgb_rfr
        comp_srgb = torch.clip(image.linear_to_srgb(comp_rgb), 0.0, 1.0)  # convert to sRGB and clip to [0, 1]
        renderings[2]['rgb'] = comp_srgb

        outputs={}

        # showed by viewer
        outputs['rgb']=renderings[2]['rgb']
        # outputs['depth']=renderings[2]['depth'].unsqueeze(-1)
        outputs['depth']=renderings_depth[2]['depth'].unsqueeze(-1)
        outputs['accumulation']=renderings[2]['acc']
        if self.config.compute_extras:
            outputs['distance_mean']=renderings[2]['distance_mean']
            outputs['distance_median']=renderings[2]['distance_median']

        # for loss calculation
        outputs['renderings']=renderings
        outputs['ray_history'] = ray_history
        outputs['ray_samples'] = ray_samples
        return outputs

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []

        def set_step(step):
            self.step = step

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_step,
            )
        )
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
            """Returns the parameter groups needed to optimizer your model components."""
            param_groups = {}
            param_groups["model"] = list(self.parameters())
            return param_groups


    def get_metrics_dict(self, outputs, batch):
        """Returns metrics dictionary which will be plotted with comet, wandb or tensorboard."""
        metrics_dict = {}
        gt_rgb = batch['image'].to(self.device)
        predicted_rgb = outputs['rgb']
        metrics_dict["psnr"] = self.psnr(gt_rgb, predicted_rgb)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Returns a dictionary of losses to be summed which will be your loss."""
        loss_dict={}
        batch['lossmult'] = torch.Tensor([1.]).to(self.device)

        data_loss, stats = train_utils.compute_data_loss(batch, outputs['renderings'], self.r3f.config)
        loss_dict['data'] = data_loss

        if self.training:
            # interlevel loss in MipNeRF360
            # if self.config.interlevel_loss_mult > 0 and not self.config.single_mlp:
            #     loss_dict['interlevel'] = train_utils.interlevel_loss(outputs['ray_history'], self.config)

            # interlevel loss in ZipNeRF360
            if self.r3f.config.anti_interlevel_loss_mult > 0 and not self.r3f.single_mlp:
                loss_dict['anti_interlevel'] = train_utils.anti_interlevel_loss(outputs['ray_history'], self.r3f.config)

            # distortion loss
            if self.r3f.config.distortion_loss_mult > 0:
                loss_dict['distortion'] = train_utils.distortion_loss(outputs['ray_history'], self.r3f.config, outputs['ray_samples'])

            # opacity loss
            # if self.config.opacity_loss_mult > 0:
            #     loss_dict['opacity'] = train_utils.opacity_loss(outputs['rgb'], self.config)

            # # orientation loss in RefNeRF
            # if (self.config.orientation_coarse_loss_mult > 0 or
            #         self.config.orientation_loss_mult > 0):
            #     loss_dict['orientation'] = train_utils.orientation_loss(batch, self.config, outputs['ray_history'],
            #                                                             self.config)
            # hash grid l2 weight decay
            if self.r3f.config.hash_decay_mults > 0:
                loss_dict['hash_decay'] = train_utils.hash_decay_loss(outputs['ray_history'], self.r3f.config)

            # # normal supervision loss in RefNeRF
            # if (self.config.predicted_normal_coarse_loss_mult > 0 or
            #         self.config.predicted_normal_loss_mult > 0):
            #     loss_dict['predicted_normals'] = train_utils.predicted_normal_loss(
            #         self.config, outputs['ray_history'], self.config)
        return loss_dict


    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]: # type: ignore
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""
        gt_rgb = batch["image"].to(self.device)

        predicted_rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        # depth = colormaps.apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])
        predicted_distance = outputs["depth"]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([predicted_distance], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        # Check for NaN in predicted_rgb
        if torch.isnan(predicted_rgb).any():
            print("Warning: predicted_rgb contains NaN values. Replacing them with zeros.")
            predicted_rgb = torch.nan_to_num(predicted_rgb, nan=0.0, posinf=1.0, neginf=0.0)

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(self.psnr(gt_rgb, predicted_rgb).item()),
            "ssim": float(self.ssim(gt_rgb, torch.clip(predicted_rgb, 0.0, 1.0))),
            "lpips": float(self.lpips(gt_rgb, torch.clip(predicted_rgb, 0.0, 1.0)))
        }

        if "mask" in batch:
            # Process each image and corresponding mask in the batch
            mask = batch["mask"].to(self.device).float()
            mask = mask.permute(2, 0, 1).unsqueeze(0)  # Reshape to [1, 1, H, W] to match the image dimensions

            # Ensure mask is not empty (i.e., contains at least one 1)
            if mask.sum() > 0:
                # Compute masked PSNR for the current image
                mask = mask.expand_as(gt_rgb)
                masked_gt_rgb = gt_rgb[mask == 1]
                masked_predicted_rgb = predicted_rgb[mask == 1]
                mse = torch.mean((masked_gt_rgb - masked_predicted_rgb) ** 2)
                masked_psnr = 10 * torch.log10((1 ** 2) / mse)
            else:
                print("Mask is empty, skipping PSNR computation for this image.")

            # Compute the average masked PSNR for the batch and store it in the metrics dictionary
            metrics_dict["masked_psnr"] = float(masked_psnr)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict
