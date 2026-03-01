import os
import sys
import numpy as np
import torch
import trimesh

from internal.mesh_preprocess import preprocess_mesh

_ext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'extensions', 'cuda')
if _ext_dir not in sys.path:
    sys.path.insert(0, _ext_dir)

import _cuda_backend


class CudaRayTracer:
    """GPU-accelerated BVH ray tracer wrapping the CUDA extension.

    Drop-in replacement for Open3D's RaycastingScene with a compatible
    cast_rays() interface, but runs entirely on the GPU.
    """

    def __init__(self, mesh_files, scale_factor=1.0,
                 convexity_threshold=0.02, smooth_iterations=10):
        all_vertices = []
        all_faces = []
        all_geo_ids = []
        vertex_offset = 0

        for mesh_idx, ply_path in enumerate(mesh_files):
            mesh = trimesh.load_mesh(ply_path)
            mesh = preprocess_mesh(
                mesh,
                convexity_threshold=convexity_threshold,
                smooth_iterations=smooth_iterations,
            )
            verts = np.array(mesh.vertices, dtype=np.float32) * scale_factor
            faces = np.array(mesh.faces, dtype=np.uint32)

            all_vertices.append(verts)
            all_faces.append(faces + vertex_offset)
            all_geo_ids.extend([mesh_idx] * len(faces))

            vertex_offset += len(verts)

        all_vertices = np.concatenate(all_vertices, axis=0).astype(np.float32)
        all_faces = np.concatenate(all_faces, axis=0).astype(np.uint32)

        assert len(all_geo_ids) == len(all_faces), "geo_ids length mismatch"
        assert all_faces.shape[0] > 8, "BVH needs at least 8 triangles."

        self.impl = _cuda_backend.create_raytracer(
            all_vertices, all_faces, all_geo_ids
        )
        self.device = None

    def cast_rays(self, origins, directions, device=None):
        """Cast rays and return results compatible with Open3D's interface.

        Args:
            origins: [N, 3] ray origins (torch.Tensor on GPU, or will be moved)
            directions: [N, 3] ray directions (torch.Tensor on GPU, or will be moved)
            device: target device (inferred from origins if None)

        Returns:
            dict with keys:
                't_hit': [N] hit distances (inf for misses, matching Open3D convention)
                'primitive_normals': [N, 3] face normals at hit points
                'geometry_ids': [N] mesh index of hit triangle (-1 for misses)
        """
        if device is None:
            device = origins.device if torch.is_tensor(origins) else torch.device('cuda')

        if not torch.is_tensor(origins):
            origins = torch.tensor(origins, dtype=torch.float32)
        if not torch.is_tensor(directions):
            directions = torch.tensor(directions, dtype=torch.float32)

        origins = origins.float().contiguous().cuda()
        directions = directions.float().contiguous().cuda()

        prefix = origins.shape[:-1]
        origins = origins.view(-1, 3)
        directions = directions.view(-1, 3)
        N = origins.shape[0]

        positions = torch.empty(N, 3, dtype=torch.float32, device='cuda')
        normals = torch.empty(N, 3, dtype=torch.float32, device='cuda')
        depth = torch.empty(N, dtype=torch.float32, device='cuda')
        geo_ids = torch.empty(N, dtype=torch.int32, device='cuda')

        self.impl.trace(origins, directions, positions, normals, depth, geo_ids)

        # Convert MAX_DIST (10.0) sentinel to inf to match Open3D convention
        depth = torch.where(depth >= 9.99, torch.tensor(float('inf'), device='cuda'), depth)

        depth = depth.view(*prefix).to(device)
        normals = normals.view(*prefix, 3).to(device)
        geo_ids = geo_ids.view(*prefix).to(device)

        return {
            't_hit': depth,
            'primitive_normals': normals,
            'geometry_ids': geo_ids,
        }
