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
Some ray datastructures.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Tuple, Union, overload, List, Any

import torch
import trimesh
import open3d as o3d
from jaxtyping import Float, Int, Shaped
from sympy.integrals.meijerint_doc import category
from torch import Tensor, nn

from internal.ray_refraction import MeshRefraction
from internal.ray_reflection import RayReflection
from nerfstudio.utils.math import Gaussians, conical_frustum_to_gaussian
from nerfstudio.utils.tensor_dataclass import TensorDataclass

from abc import abstractmethod
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union
from nerfacc import OccGridEstimator


@dataclass
class RaySamples(TensorDataclass):
    """Samples along a ray"""

    origins: Float[Tensor, "*bs 3"]
    """xyz coordinate for ray origin."""
    directions: Float[Tensor, "*bs 3"]
    """Direction of ray."""
    starts: Float[Tensor, "*bs 1"]
    """Where the frustum starts along a ray."""
    ends: Float[Tensor, "*bs 1"]
    """Where the frustum ends along a ray."""
    offsets: Optional[Float[Tensor, "*bs 3"]] = None
    """Offsets for each sample position"""
    intersections: Optional[Float[Tensor, "*bs 3"]] = None
    """Intersections between each ray and surfaces"""
    normals: Optional[Float[Tensor, "*bs 3"]] = None
    """normals at each intersection"""
    mask: Optional = None
    """mask"""
    att_coef: Optional[Float[Tensor, "*bs 1"]] = None
    """attenuation coefficient"""

    def get_positions(self) -> Float[Tensor, "*batch 3"]:
        """Calculates "center" position of frustum. Not weighted by mass.

        Returns:
            xyz positions.
        """
        pos = self.origins + self.directions * (self.starts + self.ends) / 2
        if self.offsets is not None:
            pos = pos + self.offsets
        return pos

    def get_refracted_rays(self, scene_dict: dict):
        scene = scene_dict['scene']
        iors = scene_dict['iors'].to(self.origins.device)

        # 1. Get origins, directions, r1, r2
        origins = self.origins.clone()
        directions = self.directions.clone()
        positions = self.get_positions()  # [num_rays_per_batch, num_samples_per_ray, 3]
        epsilon = 1e-4
        num_samples_per_ray = self.origins.shape[1]

        # Create some lists
        intersections_list = []
        mask_list = []
        updated_origins_list = []
        updated_directions_list = []
        indices_list = []
        indices = torch.arange(origins.shape[0], device=origins.device)

        # 2. Get intersections and normals through the first refraction
        ray_refraction = MeshRefraction(origins, directions, positions)
        intersections, normals, mask, indices, mesh_idx = ray_refraction.get_intersections_and_normals(scene, origins, directions, indices)
        normals_first = normals.clone()

        n = iors[mesh_idx]
        r = 1.0 / n # initialise r, assuming the ray starts from air
        ray_refraction.r = r  # update r for this refraction

        directions_new, mask_tir = ray_refraction.snell_fn(normals, directions)
        distance = torch.norm(origins - intersections, dim=-1)
        origins_new = intersections - directions_new * distance.unsqueeze(-1)
        updated_origins, updated_directions, updated_positions, mask_update_first = ray_refraction.update_sample_points(
            intersections, origins_new, directions_new, mask)

        intersections_list.append(intersections)
        mask_list.append(mask)
        updated_origins_list.append(updated_origins)
        updated_directions_list.append(updated_directions)
        indices_list.append(indices)

        r = r[mask[:, 0]]
        mask_in = torch.ones_like(r, dtype=torch.bool, device=origins.device)  # all elements are True, True means 'in'

        # 3. Get intersections and normals through the following refractions
        i = 0
        while True:
            ray_refraction = MeshRefraction(updated_origins[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            updated_directions[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            updated_positions[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                            r)
            intersections_offset = intersections_list[i] + directions_new * epsilon
            intersections, normals, mask, indices, mesh_idx = ray_refraction.get_intersections_and_normals(scene,
                                                                                                 intersections_offset[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                                                                                 directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3),
                                                                                                 indices_list[i])

            # flip normals where the angle between directions_new and normals is bigger than 90 degrees
            cos_theta = torch.sum(normals[:, 0, :] * directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3)[:, 0, :], dim=-1)
            normals = torch.where((cos_theta < 0)[:, None, None], normals, -normals)
            # normals = torch.where(mask_in[:, None, None], -normals, normals)

            n = iors[mesh_idx]
            r = torch.where(mask_in, n / 1.0, 1.0 / n)
            ray_refraction.r = r

            directions_new, mask_tir = ray_refraction.snell_fn(normals, directions_new[mask_list[i]].view(-1, num_samples_per_ray, 3))  # negative normals because the ray is inside the surface
            distance = distance[mask_list[i]] + torch.norm(intersections_list[i][mask_list[i]] - intersections.view(-1, 3), dim=-1)
            origins_new = intersections - directions_new * distance.view(-1, num_samples_per_ray).unsqueeze(-1)
            distance = distance.reshape(-1, num_samples_per_ray)
            updated_origins, updated_directions, updated_positions, update_mask = ray_refraction.update_sample_points(
                intersections, origins_new, directions_new, mask)

            r = r[mask[:, 0]]
            mask_in = torch.where(mask_in == mask_tir, True, False)
            mask_in = mask_in[mask[:, 0]]

            intersections_list.append(intersections)
            mask_list.append(mask)
            updated_origins_list.append(updated_origins)
            updated_directions_list.append(updated_directions)
            indices_list.append(indices)

            i += 1

            # Calculate the number of non-NaN elements in the intersections tensor
            rows_with_non_nan = ~torch.isnan(intersections).any(dim=2)
            rows_with_non_nan = rows_with_non_nan.any(dim=1)
            num_non_nan_rows = rows_with_non_nan.sum().item()

            # Break the loop if no more intersections are found
            if num_non_nan_rows == 0 or i > 30:
                break

        origins_final = origins.clone()
        directions_final = directions.clone()
        intersections = intersections_list[0].unsqueeze(0).repeat(i, 1, 1, 1)

        for j in range(i - 1):
            origins_final[indices_list[j]] = updated_origins_list[j + 1]
            directions_final[indices_list[j]] = updated_directions_list[j + 1]
            intersections[j + 1][indices_list[j]] = intersections_list[j + 1]

        self.origins = origins_final
        self.directions = directions_final
        self.intersections = intersections

        return intersections_list, normals_first, mask_update_first

    def get_reflected_rays(self, intersections, normals, masks) -> None:
        origins = self.origins.clone()
        directions = self.directions.clone()
        positions = self.get_positions()
        intersections, normals, mask = intersections.clone(), normals.clone(), masks.clone()

        # 1) Get reflective directions
        ray_reflection = RayReflection(origins, directions, positions)
        directions_new = ray_reflection.get_reflected_directions(normals)
        distance = torch.norm(origins - intersections, dim=-1)
        origins_new = intersections - directions_new * distance.unsqueeze(-1)
        updated_origins, updated_directions = ray_reflection.update_sample_points(intersections, origins_new, directions_new, mask)
        # 2) Update ray_samples.frustums.directions
        directions_final = directions.clone()
        directions_final[mask] = updated_directions[mask]
        self.directions = directions_final

        # 3) Update ray_samples.frustums.origins
        origins_final = origins.clone()
        origins_final[mask] = updated_origins[mask]
        self.origins = origins_final

        self.intersections = intersections
        self.normals = normals
        self.mask = mask
