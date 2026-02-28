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
import torch
import torch.nn.functional as F
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
from nerfstudio.utils.rich_utils import CONSOLE
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
    stage: Literal["bg", "fg", "none"] = "none"
    """Training stage: 'bg' trains background only, 'fg' trains foreground in-object field."""
    bg_checkpoint_path: str = None
    """Path to background field checkpoint for fg stage."""
    max_refracted_bounces: int = 12
    """Max refracted bounces (entry + TIR) in ray tracing; one TIR counts as one bounce (default 12)."""
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
        config.max_refracted_bounces = self.config.max_refracted_bounces

        # load mesh files and build GPU BVH ray tracer
        ior_map = {"glass": 1.5, "water": 1.333, "diamond": 2.418, "air": 1.0, "alcohol": 1.36, "plastic": 1.45, "perfume": 1.46}
        mesh_files = str(self.kwargs['ply_path']).split()
        self.scene = None
        if mesh_files and self.config.stage != "bg":
            material_list = [
                next((word for word in path.split('_') if word in ior_map), path.split('_')[-1].replace('.ply', ''))
                for path in mesh_files
            ]
            iors = [ior_map[material] for material in material_list]
            iors.append(float('nan'))
            iors = torch.tensor(iors, dtype=torch.float32)
            from internal.cuda_raytracer import CudaRayTracer
            cuda_rt = CudaRayTracer(mesh_files, scale_factor=self.kwargs['scale_factor'])
            CONSOLE.print("Using GPU BVH ray tracer for foreground stage.")
            mesh_indices = list(range(len(mesh_files)))
            # Store scalar IOR to avoid per-iteration GPU sync (.item()) in fg render.
            self.scene = {"scene": cuda_rt, "iors": iors, "ior0": float(iors[0]), "mesh_indices": mesh_indices}

        self.r3f = r3f(config=config)

        if self.config.stage == "fg":
            assert self.config.bg_checkpoint_path is not None, "bg_checkpoint_path required for fg stage"
            assert self.scene is not None, "Mesh (ply) file required for fg stage"
            self.r3f_bg = r3f(config=config)
            ckpt = torch.load(self.config.bg_checkpoint_path, map_location="cpu")
            bg_state = {
                k.replace('_model.r3f.', ''): v
                for k, v in ckpt['pipeline'].items()
                if k.startswith('_model.r3f.')
            }
            self.r3f_bg.load_state_dict(bg_state)
            self.r3f_bg.eval()
            for p in self.r3f_bg.parameters():
                p.requires_grad_(False)
            print(f"Loaded frozen BG field from {self.config.bg_checkpoint_path}")

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
        batch['directions'] = ray_bundle.directions
        batch['viewdirs'] = ray_bundle.directions
        batch['radii'] = torch.sqrt(ray_bundle.pixel_area)* 2 / torch.sqrt(torch.full_like(ray_bundle.pixel_area,12.))
        batch['cam_idx'] = ray_bundle.camera_indices
        batch['near'] = ray_bundle.nears
        batch['far'] = ray_bundle.fars
        batch['cam_dirs'] = None  # did not be calculated in raybundle
        # batch['imageplane'] = None
        # batch['exposure_values'] = None
        return batch

    def _snell_refract(self, d, n, r):
        """Snell's law refraction for per-ray tensors.
        Args:
            d: incident direction [N, 3] (unit)
            n: surface normal [N, 3] (unit, pointing towards incident side)
            r: ratio n1/n2 (scalar)
        Returns:
            refracted direction [N, 3] (unit), tir_mask [N] (True where total internal reflection)
        """
        cos_i = -(d * n).sum(-1, keepdim=True)  # [N, 1]
        sin2_t = r**2 * (1 - cos_i**2)
        tir_mask = (sin2_t.squeeze(-1) > 1.0)
        cos_t = torch.sqrt((1 - sin2_t).clamp(min=0))
        refracted = r * d + (r * cos_i - cos_t) * n
        refracted = F.normalize(refracted, dim=-1)
        reflected = d + 2 * cos_i * n
        reflected = F.normalize(reflected, dim=-1)
        result = torch.where(tir_mask.unsqueeze(-1), reflected, refracted)
        return result, tir_mask

    def _fresnel_R(self, cos_i, r):
        """Compute Fresnel reflectance (unpolarized average).
        Args:
            cos_i: cosine of incidence angle [N] (positive)
            r: ratio n1/n2 (scalar)
        Returns:
            R [N, 1]
        """
        sin2_t = r**2 * (1 - cos_i**2)
        sin2_t = sin2_t.clamp(0, 1)
        cos_t = torch.sqrt(1 - sin2_t)
        eps = 1e-6
        Rs = ((r * cos_i - cos_t) / (r * cos_i + cos_t + eps))**2
        Rp = ((r * cos_t - cos_i) / (cos_i + r * cos_t + eps))**2
        R = ((Rs + Rp) / 2).unsqueeze(-1)
        return torch.nan_to_num(R, nan=0.0)

    def _query_bg_field(self, origins, directions, radii, cam_idx, near, far):
        """Query the frozen background field for a set of rays.
        Always uses train_frac=1.0 since the bg field is fully trained."""
        bg_batch = {
            'origins': origins,
            'directions': directions,
            'viewdirs': directions,
            'radii': radii,
            'cam_idx': cam_idx,
            'near': near,
            'far': far,
            'cam_dirs': None,
        }
        renderings_bg, _, _, _, _ = self.r3f_bg(
            rand=False,
            batch=bg_batch,
            train_frac=1.0,
            compute_extras=False,
            zero_glo=True,
            straight=True,
            scene=None,
        )
        return renderings_bg[-1]['rgb']

    def _render_fg_stage(self, batch, anneal_frac):
        """Full rendering pipeline for fg stage: fg field (interior) + frozen bg field (exit/reflected).
        Only runs fg field on rays that hit the object; non-hitting rays use the frozen bg field directly."""
        device = batch['origins'].device
        cuda_rt = self.scene['scene']
        ior = self.scene.get('ior0', float(self.scene['iors'][0]))
        N = batch['origins'].shape[0]

        cam_origins = batch['origins']          # [N, 3]
        cam_dirs = batch['directions']          # [N, 3] (scaled by directions_norm)
        cam_dirs_hat = F.normalize(cam_dirs, dim=-1)

        # ── Step 1: Pre-compute entry/exit geometry via CUDA BVH ──
        with torch.no_grad():
            result_entry = cuda_rt.cast_rays(cam_origins, cam_dirs_hat, device=device)

            t_entry = result_entry['t_hit']
            entry_normals = result_entry['primitive_normals']
            hit_mask = ~torch.isinf(t_entry)
            n_hit = hit_mask.sum().item()

            entry_points = cam_origins + t_entry.unsqueeze(-1) * cam_dirs_hat

            # Orient normals towards camera
            cos_i_raw = -(cam_dirs_hat * entry_normals).sum(-1, keepdim=True)
            entry_normals = torch.where(cos_i_raw < 0, -entry_normals, entry_normals)
            cos_i = cos_i_raw.abs()   # [N, 1]

            # Reflected direction at entry
            reflected_dirs = cam_dirs_hat + 2 * cos_i * entry_normals
            reflected_dirs = F.normalize(reflected_dirs, dim=-1)

            # Refracted direction at entry (air -> object)
            refracted_dirs, _ = self._snell_refract(cam_dirs_hat, entry_normals, 1.0 / ior)

        # ── Step 2: Query frozen BG field for non-hitting camera rays ──
        final_rgb = torch.zeros(N, 3, device=device)
        with torch.no_grad():
            if n_hit < N:
                nonhit_rgb = self._query_bg_field(
                    cam_origins[~hit_mask], cam_dirs[~hit_mask],
                    batch['radii'][~hit_mask], batch['cam_idx'][~hit_mask],
                    batch['near'][~hit_mask], batch['far'][~hit_mask])
                final_rgb[~hit_mask] = torch.clip(image.linear_to_srgb(nonhit_rgb), 0.0, 1.0)

        # ── Step 3: For hitting rays, trace exit geometry and query bg field ──
        ray_history = []
        rfls = []
        ray_results = {}
        ray_samples = None
        renderings_hit = None

        if n_hit > 0:
            with torch.no_grad():
                entry_pts_h = entry_points[hit_mask]
                refr_dirs_h = refracted_dirs[hit_mask]
                n_hit = entry_pts_h.shape[0]

                # Cast refracted ray to find exit point
                eps_geom = 1e-4
                result_exit = cuda_rt.cast_rays(
                    entry_pts_h + eps_geom * refr_dirs_h, refr_dirs_h, device=device)

                t_inner = result_exit['t_hit']
                exit_normals_raw = result_exit['primitive_normals']
                exit_pts_h = entry_pts_h + (eps_geom + t_inner.unsqueeze(-1)) * refr_dirs_h

                # Orient exit normals towards interior
                cos_exit_raw = -(refr_dirs_h * exit_normals_raw).sum(-1, keepdim=True)
                exit_normals = torch.where(cos_exit_raw < 0, -exit_normals_raw, exit_normals_raw)

                # Exit direction (object -> air), with TIR bounce loop
                exit_dirs_h, tir_at_exit = self._snell_refract(refr_dirs_h, exit_normals, ior / 1.0)

                remaining_tir = tir_at_exit.clone()
                current_pts = exit_pts_h.clone()
                current_dirs = exit_dirs_h.clone()
                for _bounce in range(self.config.max_refracted_bounces - 1):  # entry = 1, TIR = rest
                    if not remaining_tir.any():
                        break
                    tir_idx = remaining_tir
                    result_bounce = cuda_rt.cast_rays(
                        current_pts[tir_idx] + eps_geom * current_dirs[tir_idx],
                        current_dirs[tir_idx], device=device)
                    t_bounce = result_bounce['t_hit']
                    bounce_normals_raw = result_bounce['primitive_normals']
                    bounce_pts = current_pts[tir_idx] + (eps_geom + t_bounce.unsqueeze(-1)) * current_dirs[tir_idx]
                    cos_b = -(current_dirs[tir_idx] * bounce_normals_raw).sum(-1, keepdim=True)
                    bounce_normals = torch.where(cos_b < 0, -bounce_normals_raw, bounce_normals_raw)
                    bounce_exit_dirs, bounce_tir = self._snell_refract(
                        current_dirs[tir_idx], bounce_normals, ior / 1.0)
                    current_pts[tir_idx] = bounce_pts
                    current_dirs[tir_idx] = bounce_exit_dirs
                    still_tir = torch.zeros_like(remaining_tir)
                    still_tir[tir_idx] = bounce_tir
                    remaining_tir = still_tir

                exit_pts_h = current_pts
                exit_dirs_h = current_dirs

                # Project exit point onto camera ray to get t_back
                t_back_h = ((exit_pts_h - cam_origins[hit_mask]) * cam_dirs_hat[hit_mask]).sum(-1)

                # Query frozen BG field for both exit and reflected rays in one batched call.
                ray_eps = 1e-3
                radii_h = batch['radii'][hit_mask]
                cam_idx_h = batch['cam_idx'][hit_mask]
                far_h = batch['far'][hit_mask]
                near_h = torch.full((n_hit, 1), 0.01, device=device)

                bg_origins = torch.cat(
                    [exit_pts_h + ray_eps * exit_dirs_h, entry_pts_h + ray_eps * reflected_dirs[hit_mask]],
                    dim=0,
                )
                bg_dirs = torch.cat([exit_dirs_h, reflected_dirs[hit_mask]], dim=0)
                bg_radii = torch.cat([radii_h, radii_h], dim=0)
                bg_cam_idx = torch.cat([cam_idx_h, cam_idx_h], dim=0)
                bg_near = torch.cat([near_h, near_h], dim=0)
                bg_far = torch.cat([far_h, far_h], dim=0)

                bg_rgb = self._query_bg_field(bg_origins, bg_dirs, bg_radii, bg_cam_idx, bg_near, bg_far)
                exit_rgb_h, reflected_rgb_h = bg_rgb[:n_hit], bg_rgb[n_hit:]

            # ── Step 4: Constrain near/far and build hitting-only batch ──
            near_offset = 0.05
            far_offset = 0.05
            hit_near = (t_entry[hit_mask] - near_offset).clamp(min=1e-3).unsqueeze(-1)
            hit_far = torch.minimum((t_back_h + far_offset), batch['far'].max()).unsqueeze(-1)
            hit_far = torch.max(hit_far, hit_near + 0.02)

            batch_hit = {
                'origins': batch['origins'][hit_mask],
                'directions': batch['directions'][hit_mask],
                'viewdirs': batch['viewdirs'][hit_mask],
                'radii': batch['radii'][hit_mask],
                'cam_idx': batch['cam_idx'][hit_mask],
                'near': hit_near,
                'far': hit_far,
                'cam_dirs': None,
            }

            # ── Step 5: Run FG field on hitting rays only ──
            renderings_hit, ray_history, rfls, ray_results, ray_samples = self.r3f(
                rand=self.config.rand if self.training else False,
                batch=batch_hit,
                train_frac=anneal_frac,
                compute_extras=self.config.compute_extras,
                zero_glo=self.config.zero_glo if self.training else True,
                reflection=None,
                scene=self.scene,
                custom_bg_rgbs=exit_rgb_h,
                t_max=t_back_h + far_offset,
            )

            # ── Step 6: Fresnel combine (linear space, hitting rays only) ──
            # interior_rgb_h = exit_rgb_h  # DEBUG: bypass fg field, use bg exit color directly (density=0 equivalent)
            interior_rgb_h = renderings_hit[-1]['rgb']
            R_h = self._fresnel_R(cos_i[hit_mask].squeeze(-1), 1.0 / ior)
            comp_h = R_h * reflected_rgb_h + (1 - R_h) * interior_rgb_h
            final_rgb[hit_mask] = torch.clip(image.linear_to_srgb(comp_h), 0.0, 1.0)

        # ── Step 7: Assemble full-size renderings for loss computation ──
        renderings = []
        if renderings_hit is not None:
            for rend_h in renderings_hit:
                full_rend = {}
                for k, v in rend_h.items():
                    if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == n_hit:
                        full_v = torch.zeros(N, *v.shape[1:], device=device, dtype=v.dtype)
                        full_v[hit_mask] = v
                        full_rend[k] = full_v
                    else:
                        full_rend[k] = v
                renderings.append(full_rend)
            renderings[-1]['rgb'] = final_rgb
        else:
            renderings = [{'rgb': final_rgb, 'depth': torch.zeros(N, device=device),
                           'acc': torch.zeros(N, device=device)}]

        return renderings, ray_history, rfls, ray_results, ray_samples, hit_mask

    def get_outputs(self, ray_bundle: RayBundle):
        ray_bundle.metadata["viewdirs"] = ray_bundle.directions
        ray_bundle.metadata["radii"] = torch.sqrt(ray_bundle.pixel_area)* 2 / torch.sqrt(torch.full_like(ray_bundle.pixel_area,12.))
        ray_bundle.directions = ray_bundle.directions * ray_bundle.metadata["directions_norm"]

        if self.training:
            anneal_frac = np.clip(self.step / self.config.proposal_weights_anneal_max_num_iters, 0, 1)
        else:
            anneal_frac = 1.0
        batch = self.construct_batch_from_raybundle(ray_bundle)

        if self.config.stage == "bg":
            # Background-only stage: standard NeRF without refraction/reflection
            renderings, ray_history, rfls, ray_results, ray_samples = self.r3f(
                rand=self.config.rand if self.training else False,
                batch=batch,
                train_frac=anneal_frac,
                compute_extras=self.config.compute_extras,
                zero_glo=self.config.zero_glo if self.training else True,
                reflection=None,
                scene=None,
                straight=True,
            )
            renderings[-1]['rgb'] = torch.clip(image.linear_to_srgb(renderings[-1]['rgb']), 0.0, 1.0)

        elif self.config.stage == "fg":
            renderings, ray_history, rfls, ray_results, ray_samples, hit_mask = \
                self._render_fg_stage(batch, anneal_frac)

        else:
            renderings, ray_history, rfls, ray_results, ray_samples = self.r3f(
                    rand=self.config.rand if self.training else False,
                    batch=batch,
                    train_frac=anneal_frac,
                    compute_extras=self.config.compute_extras,
                    zero_glo=self.config.zero_glo if self.training else True,
                    reflection=None,
                    scene=self.scene,)

            renderings_rfl, _, _, _, _ = self.r3f(
                rand=self.config.rand if self.training else False,
                batch=batch,
                train_frac=anneal_frac,
                compute_extras=self.config.compute_extras,
                zero_glo=self.config.zero_glo if self.training else True,
                reflection=rfls,)

            # Fresnel equation
            ray_samples_rfl = rfls[-1]
            normals = ray_samples_rfl.normals
            ray_reflection = RayReflection(ray_samples_rfl.origins, ray_samples_rfl.directions, ray_samples_rfl.get_positions(), 1/self.scene['iors'][0])
            R = ray_reflection.fresnel_fn(normals)  # [8192, 1]
            rgb_rfl, rgb_rfr = renderings_rfl[2]['rgb'], renderings[2]['rgb']
            comp_rgb = R * rgb_rfl + (1 - R) * rgb_rfr
            comp_srgb = torch.clip(image.linear_to_srgb(comp_rgb), 0.0, 1.0)
            renderings[2]['rgb'] = comp_srgb

        outputs={}

        # showed by viewer
        outputs['rgb']=renderings[-1]['rgb']
        depth = renderings[-1]['depth']
        outputs['depth']=depth.unsqueeze(-1) if depth.ndim == 1 else depth
        outputs['accumulation']=renderings[-1]['acc']
        if self.config.compute_extras:
            outputs['distance_mean']=renderings[-1].get('distance_mean', depth)
            outputs['distance_median']=renderings[-1].get('distance_median', depth)

        # for loss calculation
        outputs['renderings']=renderings
        outputs['ray_history'] = ray_history
        outputs['ray_samples'] = ray_samples
        if self.config.stage == "fg":
            outputs['hit_mask'] = hit_mask
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
            param_groups["model"] = [p for p in self.parameters() if p.requires_grad]
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

        if self.config.stage == "bg" and "fg_mask" in batch:
            fg_mask = batch["fg_mask"].to(self.device).float()
            batch['lossmult'] = (1.0 - fg_mask)[..., 0:1]
        elif self.config.stage == "fg" and "hit_mask" in outputs:
            hit_mask = outputs["hit_mask"].to(self.device).float()
            batch['lossmult'] = hit_mask.unsqueeze(-1)
        else:
            batch['lossmult'] = torch.Tensor([1.]).to(self.device)

        data_loss, stats = train_utils.compute_data_loss(batch, outputs['renderings'], self.r3f.config)
        loss_dict['data'] = data_loss

        if self.training:
            # interlevel loss in ZipNeRF360
            if self.r3f.config.anti_interlevel_loss_mult > 0 and not self.r3f.single_mlp:
                loss_dict['anti_interlevel'] = train_utils.anti_interlevel_loss(outputs['ray_history'], self.r3f.config)

            # distortion loss (bg only; disabled for fg stage)
            if self.r3f.config.distortion_loss_mult > 0 and self.config.stage == "bg":
                loss_dict['distortion'] = train_utils.distortion_loss_bg(outputs['ray_history'], self.r3f.config)

            # hash grid l2 weight decay
            if self.r3f.config.hash_decay_mults > 0:
                loss_dict['hash_decay'] = train_utils.hash_decay_loss(outputs['ray_history'], self.r3f.config)

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

        # Move images to CPU and ensure [0, 1] for wandb/tensorboard (avoids black images from GPU tensors)
        for key in ("img", "accumulation"):
            images_dict[key] = images_dict[key].detach().cpu().clamp(0.0, 1.0)
        depth_val = images_dict["depth"]
        images_dict["depth"] = depth_val.detach().cpu()
        d = images_dict["depth"]
        if d.numel() > 0 and d.max() > d.min():
            images_dict["depth"] = (d - d.min()) / (d.max() - d.min() + 1e-8)

        return metrics_dict, images_dict
