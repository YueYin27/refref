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

"""Data parser for blender dataset"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class BlenderDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: Blender)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: Optional[str] = "white"
    """alpha color of background, when set to None, InputDataset that consumes DataparserOutputs will not attempt 
    to blend with alpha_colors using image's alpha channel data. Thus rgba image will be directly used in training. """
    ply_path: Optional[Path] = None
    """Path to PLY file to load 3D points from, defined relative to the dataset directory. This is helpful for
    Gaussian splatting and generally unused otherwise. If `None`, points are initialized randomly."""
    split_override: Optional[str] = None
    """If set, every split loads transforms_{split_override}.json instead of its own. Use "vis" to render the
    fly-through trajectory the synthetic scenes ship, which is not one of the train/val/test splits."""
    use_distortion: bool = True
    """Whether to pass the OPENCV distortion coefficients (k1..k4, p1, p2) in transforms.json to the cameras, so
    nerfstudio undistorts rays as it does for the other baselines. The real captures carry real distortion
    (k1~0.23, k2~-0.65, k3~0.49: ~10px at the image corner, ~1px over the object). Set False to reproduce the
    pinhole model NU-NeRF used when it extracted the meshes, if the traced mesh silhouette looks offset."""


@dataclass
class Blender(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: BlenderDataParserConfig

    def __init__(self, config: BlenderDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        if self.alpha_color is not None:
            self.alpha_color_tensor = get_color(self.alpha_color)
        else:
            self.alpha_color_tensor = None
        self.scale_factor = config.scale_factor
        self.ply_path = config.ply_path

    def _find_mask(self, image_path: Path) -> Optional[Path]:
        """Locate the segmentation mask for an image that transforms.json does not name.

        The synthetic scenes carry a "mask_file_path" per frame, but the real captures were
        segmented (utils/sam3_infer.py) after their JSON was written, and keep the masks in
        masks/ under the image's own name. The bg stage cannot run without them -- it is
        defined by dropping the object from the loss -- so look them up by name.
        """
        mask_dir = self.data / "masks"
        if not mask_dir.is_dir():
            return None
        for suffix in (image_path.suffix, ".png", ".jpg", ".jpeg"):
            candidate = mask_dir / (image_path.stem + suffix)
            if candidate.is_file():
                return candidate
        return None

    def _generate_dataparser_outputs(self, split="train"):
        if self.config.split_override is not None:
            split = self.config.split_override
        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []
        # Intrinsics are per frame, not per scene: the synthetic scenes share one
        # camera_angle_x, while the real captures put fl_x/fl_y/cx/cy (and the distortion
        # coefficients) at the top level, and nerfstudio's own format allows either. Read
        # each frame with the top-level entry as its fallback rather than assuming one lens.
        fx, fy, cx, cy, heights, widths, distortion_params = [], [], [], [], [], [], []

        def _intrinsic(frame, key, default=None):
            value = frame.get(key, meta.get(key, default))
            return None if value is None else float(value)

        for frame in meta["frames"]:
            # Blender exports leave the extension off ("./train/r_0"); the real captures
            # keep it ("images/frame_00001.jpg").
            rel_path = Path(frame["file_path"].replace("./", ""))
            fname = self.data / (rel_path if rel_path.suffix else rel_path.with_suffix(".png"))
            image_filenames.append(fname)
            if "mask_file_path" in frame:
                mask_filenames.append(self.data / Path(frame["mask_file_path"].replace("./", "")))
            else:
                mask_fname = self._find_mask(fname)
                if mask_fname is not None:
                    mask_filenames.append(mask_fname)
            if "depth_file_path" in frame:
                depth_filenames.append(self.data / Path(frame["depth_file_path"].replace("./", "")))
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)
        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]

        for frame in meta["frames"]:
            width = int(_intrinsic(frame, "w", image_width))
            height = int(_intrinsic(frame, "h", image_height))
            focal_length_x = _intrinsic(frame, "fl_x")
            if focal_length_x is None:
                # Blender export: one horizontal FOV, square pixels, principal point centred.
                camera_angle_x = float(_intrinsic(frame, "camera_angle_x"))
                focal_length_x = 0.5 * width / np.tan(0.5 * camera_angle_x)
                focal_length_y = focal_length_x
            else:
                focal_length_y = _intrinsic(frame, "fl_y", focal_length_x)
            widths.append(width)
            heights.append(height)
            fx.append(focal_length_x)
            fy.append(focal_length_y)
            cx.append(_intrinsic(frame, "cx", width / 2.0))
            cy.append(_intrinsic(frame, "cy", height / 2.0))
            distortion_params.append(
                camera_utils.get_distortion_params(
                    k1=_intrinsic(frame, "k1", 0.0),
                    k2=_intrinsic(frame, "k2", 0.0),
                    k3=_intrinsic(frame, "k3", 0.0),
                    k4=_intrinsic(frame, "k4", 0.0),
                    p1=_intrinsic(frame, "p1", 0.0),
                    p2=_intrinsic(frame, "p2", 0.0),
                )
            )

        distortion = torch.stack(distortion_params)
        # An all-zero stack is what the synthetic scenes produce; leave it as None there so
        # nerfstudio skips the undistortion path entirely rather than running it as a no-op.
        if not self.config.use_distortion or not torch.any(distortion):
            distortion = None

        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=torch.tensor(fx, dtype=torch.float32),
            fy=torch.tensor(fy, dtype=torch.float32),
            cx=torch.tensor(cx, dtype=torch.float32),
            cy=torch.tensor(cy, dtype=torch.float32),
            height=torch.tensor(heights, dtype=torch.int32),
            width=torch.tensor(widths, dtype=torch.int32),
            distortion_params=distortion,
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) == len(image_filenames) else None
            }
        if self.config.ply_path is not None:
            metadata.update(self._load_3D_points(self.config.data / self.config.ply_path))

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            mask_filenames=mask_filenames if len(mask_filenames) == len(image_filenames) else None,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata=metadata,
        )

        return dataparser_outputs

    def _load_3D_points(self, ply_file_path: Path):
        import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        pcd = o3d.io.read_point_cloud(str(ply_file_path))

        points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32) * self.config.scale_factor)
        points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }
        return out
