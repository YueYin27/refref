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
Dataset.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from nerfstudio.data.datasets.base_dataset import InputDataset

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from internal.data_utils import get_image_mask_tensor_from_path, get_depth_image_from_path


class RefRefDataset(InputDataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask", "fg_mask"]
    cameras: Cameras

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, mode: Literal["train", "eval"] = "train", stage: Literal["bg", "fg", "none"] = "none"):
        super().__init__(dataparser_outputs=dataparser_outputs)
        print("✅ RefRefDataset: Initialised with", len(dataparser_outputs.image_filenames), "images in", mode, "mode, stage:", stage)
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.mask_color = dataparser_outputs.metadata.get("mask_color", None)
        self.mode = mode
        self.stage = stage

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)

        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        # Handle greyscale images: convert to 3 channels by copying the single channel
        if len(image.shape) == 2:
            # Greyscale image (H, W) -> (H, W, 3)
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Greyscale image with channel dimension (H, W, 1) -> (H, W, 3)
            image = np.repeat(image, 3, axis=-1)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image_float32(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in float32 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_image_uint8(self, image_idx: int) -> UInt8[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in uint8 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx))
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * (image[:, :, -1:] / 255.0) + 255.0 * self._dataparser_outputs.alpha_color * (
                1.0 - image[:, :, -1:] / 255.0
            )
            image = torch.clamp(image, min=0, max=255).to(torch.uint8)
        return image

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data = {"image_idx": image_idx, "image": image}

        # In bg stage, load foreground mask during training so we can exclude object rays from the loss
        if self.stage == "bg" and self.mode == "train":
            if self._dataparser_outputs.mask_filenames is not None:
                mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
                fg_mask = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
                assert (
                    fg_mask.shape[:2] == data["image"].shape[:2]
                ), f"Mask and image have different shapes. Got {fg_mask.shape[:2]} and {data['image'].shape[:2]}"
                data["fg_mask"] = fg_mask.float()

        if self.mode == "eval":
            if self._dataparser_outputs.mask_filenames is not None:
                mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
                data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)

                assert (
                    data["mask"].shape[:2] == data["image"].shape[:2]
                ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
            if self._dataparser_outputs.metadata is not None:
                depth_filepath = self._dataparser_outputs.metadata["depth_filenames"][image_idx]
                data["depth"] = get_depth_image_from_path(filepath=depth_filepath, height=800, width=800, scale_factor=self.scale_factor)
            if self.mask_color:
                data["image"] = torch.where(
                    data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
                )
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        del data
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self._dataparser_outputs.image_filenames
