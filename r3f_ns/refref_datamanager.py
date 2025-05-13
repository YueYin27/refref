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
R3F DataManager
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from r3f_ns.refref_dataset import RefRefDataset
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class RefRefDataManagerConfig(VanillaDataManagerConfig):
    """R3F DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: RefRefDataManager)


class RefRefDataManager(VanillaDataManager[RefRefDataset]):
    """R3F DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """
    train_dataset = RefRefDataset
    eval_dataset = RefRefDataset
    config: RefRefDataManagerConfig

    def __init__(
        self,
        config: RefRefDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):

        self.config = config
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()

        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

        self.setup_train()
        self.setup_eval()

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        batch['rgb'] = batch['image'].to(self.device)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        batch['rgb'] = batch['image'].to(self.device)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def create_train_dataset(self) -> RefRefDataset:
        CONSOLE.print("Creating RefRef train dataset...")
        return RefRefDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split="train"),
            scale_factor=self.config.camera_res_scale_factor,
            mode="train",
        )

    def create_eval_dataset(self) -> RefRefDataset:
        CONSOLE.print("Creating RefRef eval dataset...")
        return RefRefDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            mode="eval",
        )

    def setup_train(self):
        """Setup the training dataset and dataloader for R3F."""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up RefRef training dataset...")
        # Initialize your custom dataset (HFDataset)
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

        # # Setup your DataLoader using the custom dataset
        # self.train_dataloader = torch.utils.data.DataLoader(
        #     self.train_dataset,
        #     batch_size=self.config.batch_size,
        #     shuffle=True,  # Or based on your requirements
        #     num_workers=self.config.num_workers,
        # )

    def setup_eval(self):
        """Setup the evaluation dataset and dataloader for R3F."""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up RefRef evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        # # Initialize your custom dataset (HFDataset) for evaluation
        # eval_dataparser_outputs = self._get_eval_dataparser_outputs()  # Adjust this based on your logic
        # self.eval_dataset = HFDataset(
        #     dataparser_outputs=eval_dataparser_outputs,
        #     scale_factor=self.config.scale_factor,  # Adjust if you have this in your config
        #     mode="eval"
        # )
        #
        # # Setup your DataLoader using the custom dataset
        # self.eval_dataloader = torch.utils.data.DataLoader(
        #     self.eval_dataset,
        #     batch_size=self.config.batch_size,
        #     shuffle=False,  # Or based on your requirements
        #     num_workers=self.config.num_workers,
        # )
