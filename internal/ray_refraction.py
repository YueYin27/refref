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
        self.directions = directions / torch.norm(directions, p=2, dim=-1, keepdim=True)
        self.positions = positions
        self.r = r
        # #region agent log
        import json as _json_dbg5; _norms_l2 = torch.norm(directions, p=2, dim=-1); _norms_minf = torch.norm(directions, p=-1, dim=-1); _norms_after = torch.norm(self.directions, p=2, dim=-1); open('/workspace/.cursor/debug-a2f859.log','a').write(_json_dbg5.dumps({"sessionId":"a2f859","hypothesisId":"H4","location":"ray_refraction.py:MeshRefraction.__init__","message":"Direction norm check","data":{"input_l2_mean":float(_norms_l2.mean().item()),"input_minf_mean":float(_norms_minf.mean().item()),"after_l2_mean":float(_norms_after.mean().item()),"after_l2_min":float(_norms_after.min().item()),"after_l2_max":float(_norms_after.max().item())}})+'\n')
        # #endregion

    def get_intersections_and_normals(self, scene, origins, directions, indices_prev):
        """
        Get intersections and surface normals using GPU BVH ray tracer.

        Args:
            scene: CudaRayTracer instance
            origins: [num_rays, num_samples, 3] ray origins
            directions: [num_rays, num_samples, 3] ray directions
            indices_prev: indices of active rays from previous iteration
        """
        device = self.origins.device
        prefix_shape = origins.shape[:-1]  # [num_rays, num_samples]

        # Flatten to [N, 3] for the CUDA raytracer
        origins_flat = origins.reshape(-1, 3)
        directions_flat = directions.reshape(-1, 3)

        results = scene.cast_rays(origins_flat, directions_flat, device=device)

        # Reshape results back to [num_rays, num_samples, ...]
        t_hit = results['t_hit'].reshape(*prefix_shape).unsqueeze(-1)
        normals = results['primitive_normals'].reshape(*prefix_shape, 3)
        mesh_idx = results['geometry_ids'].reshape(*prefix_shape)[:, 0]

        intersections = origins + t_hit * directions

        # Check for valid intersections (not inf)
        mask = ~torch.isinf(t_hit).any(dim=-1)
        mask[:, :] = mask[:, 0].unsqueeze(1).expand(-1, mask.shape[1])
        intersections = torch.where(mask.unsqueeze(-1), intersections, torch.tensor(float('nan'), device=device))
        normals = torch.where(mask.unsqueeze(-1), normals, torch.tensor(float('nan'), device=device))
        mesh_idx = torch.where(mask[:, 0], mesh_idx, torch.tensor(-1, dtype=mesh_idx.dtype, device=device))

        indices = torch.nonzero(torch.all(mask, dim=-1)).squeeze(dim=-1)
        indices = indices_prev[indices]

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

        # Adjust r's shape for broadcasting (match original scalar Snell fn behavior)
        sqrt_term = 1 - (r[:, None] ** 2) * (1 - c ** 2)
        total_internal_reflection_mask = sqrt_term < 1e-6
        # #region agent log
        import json as _json_dbg6; open('/workspace/.cursor/debug-a2f859.log','a').write(_json_dbg6.dumps({"sessionId":"a2f859","hypothesisId":"H3+H4","location":"ray_refraction.py:MeshRefraction.snell_fn","message":"snell_fn sqrt_term and TIR","data":{"r_min":float(r.min().item()),"r_max":float(r.max().item()),"c_min":float(c.min().item()),"c_max":float(c.max().item()),"sqrt_min":float(sqrt_term.min().item()),"sqrt_max":float(sqrt_term.max().item()),"tir_count":int(total_internal_reflection_mask.sum().item()),"total_elements":int(total_internal_reflection_mask.numel())}})+'\n')
        # #endregion

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
