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


"""Data parser for Hugging Face dataset repository"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, Literal

import imageio
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color

try:
    from datasets import load_dataset
    from huggingface_hub import login
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class RefRefDataParserConfig(DataParserConfig):
    """Hugging Face dataset parser config"""

    _target: Type = field(default_factory=lambda: RefRefDataParser)
    """target class to instantiate"""
    hf_repo: str = "yinyue27/RefRef"
    """Hugging Face repository name"""
    scene_name: str = "cube_smcvx_ampoule"
    """Specific scene/split to load from the dataset"""
    data_split: Literal["train", "val", "test"] = "train"
    """Which subset to use within the scene (train=first 100, val=next 100, test=last 100)"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: Optional[str] = "white"
    """alpha color of background"""
    ply_path: Optional[Path] = None
    """Path to PLY file to load 3D points from"""
    default_scene_box: bool = True
    """Whether to use default scene box dimensions (7.3 units)"""


@dataclass
class RefRefDataParser(DataParser):
    """Hugging Face Dataset Parser for RefRef repository"""

    config: RefRefDataParserConfig
    local_data_dir: Path = Path("data/RefRef_dataset_cache")

    def __init__(self, config: RefRefDataParserConfig):
        super().__init__(config=config)

        if not HF_AVAILABLE:
            raise ImportError("Please install datasets package: pip install datasets")

        # Initialise instance variables
        self.hf_repo = config.hf_repo
        self.scene_name = config.scene_name
        self.data_split = config.data_split
        self.scale_factor = config.scale_factor
        self.alpha_color = config.alpha_color
        self.ply_path = config.ply_path
        self.default_scene_box = config.default_scene_box
        self.local_data_dir.mkdir(parents=True, exist_ok=True)

        # Parse subset
        parts = self.scene_name.split('_')
        if len(parts) < 2:
            raise ValueError(f"Invalid scene_name format: {self.scene_name}")

        prefix, middle = parts[0], parts[1]
        if prefix in ["cube", "env", "sphere"]:
            if middle.startswith("mmncvx"):
                self.subset = f"{prefix}bg_multiple-non-convex"
            elif middle.startswith("smcvx"):
                self.subset = f"{prefix}bg_single-convex"
            elif middle.startswith("smncvx"):
                self.subset = f"{prefix}bg_single-non-convex"
            else:
                raise ValueError(f"Unknown middle part '{middle}' in scene_name {self.scene_name}")
        else:
            raise ValueError(f"Unknown prefix '{prefix}' in scene_name {self.scene_name}")

        if self.alpha_color is not None:
            self.alpha_color_tensor = get_color(self.alpha_color)
        else:
            self.alpha_color_tensor = None

        # Load full dataset split and save files locally
        self.image_paths = []
        self.depth_paths = []
        self.mask_paths = []
        self.poses = []

        dataset = load_dataset(
            self.hf_repo,
            name=self.subset,
            split=self.scene_name,
            verification_mode="no_checks"
        )

        scene_dir = self.local_data_dir / self.scene_name
        for idx, frame in enumerate(dataset):
            split = self._get_split_from_index(idx)

            # Create subdir and filename
            split_dir = scene_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            img_path = split_dir / f"image_{idx:04d}.png"
            depth_path = split_dir / f"depth_{idx:04d}.png"
            mask_path = split_dir / f"mask_{idx:04d}.png"

            if not img_path.exists():
                frame["image"].save(img_path)
            if not depth_path.exists():
                frame["depth"].save(depth_path)
            if not mask_path.exists():
                frame["mask"].save(mask_path)

            self.image_paths.append(img_path)
            self.depth_paths.append(depth_path)
            self.mask_paths.append(mask_path)
            self.poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

    def _get_split_from_index(self, idx: int) -> str:
        if idx < 100:
            return "train"
        elif idx < 200:
            return "val"
        else:
            return "test"

    def _generate_dataparser_outputs(self, split="train"):

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid data_split: {split}")

        if split == "train":
            indices = range(0, 100)
        elif split == "val":
            indices = range(100, 200)
        else:
            indices = range(200, 300)

        image_filenames = [self.image_paths[i] for i in indices]
        depth_filenames = [self.depth_paths[i] for i in indices]
        mask_filenames = [self.mask_paths[i] for i in indices]
        poses = np.array([self.poses[i] for i in indices])

        image_height, image_width = imageio.v2.imread(image_filenames[0]).shape[:2]
        camera_angle_x = 1.1656107902526855
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0

        camera_to_world = torch.from_numpy(poses[:, :3])
        camera_to_world[..., 3] *= self.scale_factor

        aabb = 7.3 * self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-aabb, -aabb, -aabb], [aabb, aabb, aabb]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {
            "depth_filenames": depth_filenames,
            "scene_name": self.scene_name,
            "data_split": split,
        }

        if self.ply_path and self.ply_path.exists():
            metadata.update(self._load_3D_points(self.ply_path))

        return DataparserOutputs(
            image_filenames=image_filenames,
            mask_filenames=mask_filenames,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata=metadata,
        )


    def _load_3D_points(self, ply_path: Path):
        """Load 3D points from PLY file"""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(ply_path))
            points = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32) * self.scale_factor)
            colors = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))
            return {
                "points3D_xyz": points,
                "points3D_rgb": colors,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load PLY file: {e}")