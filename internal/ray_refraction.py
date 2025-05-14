# Copyright 2025 Yue Yin and Enze Tao (The Australian National University).
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


import os

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch


class RayRefraction:
    """Ray refracting

    Args:
        origins: camera ray origins
        directions: original directions of camera rays
        positions: original positions of sample points
        r: n1/n2
    """

    def __init__(self, origins, directions, positions, r=None, radius=None):
        self.origins = origins
        self.directions = directions / torch.norm(directions, p=2, dim=-1, keepdim=True)
        self.positions = positions
        self.r = r
        self.radius = radius

    def get_intersections_and_normals(self, condition):
        raise NotImplementedError

    # @functools.lru_cache(maxsize=128)
    def snell_fn(self, n):
        """Get new ray directions based on Snell's Law, including handling total internal reflection.

        Args:
            n: surface normals
        Returns:
            refracted directions or reflected directions in case of total internal reflection
        """

        r = self.r
        l = self.directions / torch.norm(self.directions, p=2, dim=-1,
                                         keepdim=True)  # Normalize ray directions
        c = -torch.einsum('ijk, ijk -> ij', n, l)  # Cosine of the angle between the surface normal and ray direction
        sqrt_term = 1 - (r ** 2) * (1 - c ** 2)
        total_internal_reflection_mask = sqrt_term < 1e-6  # Check for total internal reflection (sqrt_term <= 0)
        flag = total_internal_reflection_mask.any()  # the flag is a boolean value

        # Refracted directions for non-total-reflection cases
        refracted_directions = r * l + (r * c - torch.sqrt(torch.clamp(sqrt_term, min=0))).unsqueeze(-1) * n
        refracted_directions = torch.nn.functional.normalize(refracted_directions, dim=-1)

        # Total internal reflection case
        reflected_directions = l + 2 * c.unsqueeze(-1) * n
        reflected_directions = torch.nn.functional.normalize(reflected_directions, dim=-1)

        # Return refracted directions where there's no total internal reflection, otherwise return reflected directions
        result_directions = torch.where(total_internal_reflection_mask.unsqueeze(-1), reflected_directions,
                                        refracted_directions)

        return result_directions, flag

    def update_sample_points(self, intersections, directions_new, condition, mask):
        raise NotImplementedError


class MeshRefraction(RayRefraction):

    def __init__(self, origins, directions, positions, r=None, ):
        # super().__init__(origins, directions, positions, r)
        self.origins = origins
        self.directions = directions / torch.norm(directions, p=-1, dim=-1, keepdim=True)
        self.positions = positions
        self.r = r

    # @functools.lru_cache(maxsize=128)
    def get_intersections_and_normals(self, scene, origins, directions, indices_prev):
        """
        Get intersections and surface normals

        Args:
            scene: the scene of the 3D object
        """
        device = self.origins.device

        # Prepare rays
        rays = torch.cat((origins, directions), dim=-1).cpu().numpy()  # Prepare rays in the required format
        rays_o3d = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

        # Cast rays
        results = scene.cast_rays(rays_o3d)  # Cast rays

        # Convert results to PyTorch tensors and move to the correct device
        t_hit = torch.tensor(results['t_hit'].cpu().numpy(), device=device).unsqueeze(-1)
        intersections = origins + t_hit * directions
        normals = torch.tensor(results["primitive_normals"].cpu().numpy(), device=device)
        mesh_idx = torch.tensor(results["geometry_ids"].cpu().numpy().astype(np.int32), device=device)[:, 0]

        # check if the intersection is not 'inf' and create a mask for valid intersections and normals
        mask = ~torch.isinf(t_hit).any(dim=-1)
        mask[:, :] = mask[:, 0].unsqueeze(1).expand(-1, mask.shape[1])  # make sure each ray has the same mask
        intersections = torch.where(mask.unsqueeze(-1), intersections, torch.tensor(float('nan'), device=device))  # mask the invalid intersections
        normals = torch.where(mask.unsqueeze(-1), normals, torch.tensor(float('nan'), device=device))  # mask the invalid normals
        mesh_idx = torch.where(mask[:, 0], mesh_idx, torch.tensor(-1, device=device))  # mask the invalid mesh indices

        # Create an indices tensor to store the indices of true values in mask
        indices = torch.nonzero(torch.all(mask, dim=-1)).squeeze(dim=-1)  # the indices of the True values, [num_of_rays]
        indices = indices_prev[indices]  # [num_of_rays], the indices of the previous True values

        return intersections, normals, mask, indices, mesh_idx

    def update_sample_points(self, intersections, origins_new, directions_new, mask):
        """Update sample points

        Args:
            intersections: intersections of the camera ray with the surface of the object
            origins_new: refracted origins
            directions_new: refracted directions
            mask: -
        """
        # 1. Calculate Euclidean distances
        distances = torch.norm(intersections - self.positions, dim=-1)

        # Get the indices of the two smallest distances along axis -1
        top2_indices = torch.topk(distances, 2, largest=False, dim=-1).indices
        top1_idx = top2_indices[:, 0]
        top2_idx = top2_indices[:, 1]
        first_idx = torch.max(top1_idx, top2_idx)  # get the latter index

        # 2. Get the mask of all samples to be updated
        first_idx = first_idx.unsqueeze(1)
        mask = (~torch.isnan(intersections).any(dim=2) & ~torch.isnan(directions_new).any(dim=2)) & mask
        mask = mask & (torch.arange(self.positions.shape[1], device=self.positions.device).unsqueeze(0) >= first_idx)

        # 3. Move the original sample points onto the refracted ray
        distances_to_intersection = torch.norm(self.positions - intersections, dim=-1)
        distances_to_intersection[~mask] = float('nan')
        updated_positions = intersections + distances_to_intersection.unsqueeze(2) * directions_new
        self.positions[mask] = updated_positions[mask]
        self.origins[mask] = origins_new[mask].clone()
        self.directions[mask] = directions_new[mask].clone()

        self.directions = torch.nn.functional.normalize(self.directions, p=2, dim=-1)

        return self.origins, self.directions, self.positions, mask

    def snell_fn(self, n, l):
        """Get new ray directions based on Snell's Law, including handling total internal reflection.

        Args:
            n: surface normals
            l: ray directions
        Returns:
            refracted directions or reflected directions in case of total internal reflection
        """

        r = self.r
        l = l / torch.norm(l, p=2, dim=-1, keepdim=True)  # Normalize ray directions
        c = -torch.einsum('ijk, ijk -> ij', n, l)  # Cosine between normals and directions

        # Adjust r's shape for broadcasting
        sqrt_term = 1 - (r[:, None] ** 2) * (1 - c ** 2)
        total_internal_reflection_mask = sqrt_term <= 0

        # create a mask to check if this is a total internal reflection
        tir_mask = total_internal_reflection_mask.any(dim=-1)  # True if there is TIR

        # Refracted directions for non-total-internal-reflection cases
        refracted_directions = r[:, None, None] * l + (r[:, None] * c - torch.sqrt(torch.clamp(sqrt_term, min=0))
                                                      )[:, :, None] * n
        refracted_directions = torch.nn.functional.normalize(refracted_directions, dim=-1)

        # Total internal reflection case
        reflected_directions = l + 2 * c.unsqueeze(-1) * n
        reflected_directions = torch.nn.functional.normalize(reflected_directions, dim=-1)

        # Return refracted directions where there's no total internal reflection, otherwise return reflected directions
        result_directions = torch.where(total_internal_reflection_mask.unsqueeze(-1), reflected_directions,
                                        refracted_directions)

        return result_directions, tir_mask
