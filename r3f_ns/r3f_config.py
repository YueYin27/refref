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
Nerfstudio R3F Config

"""

from __future__ import annotations

from r3f_ns.refref_datamanager import (
    RefRefDataManagerConfig,
)
from r3f_ns.r3f_model import R3FModelConfig
from r3f_ns.r3f_pipeline import (
    R3FPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
# from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from r3f_ns.refref_dataparser import RefRefDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


r3f_method = MethodSpecification(
    config=TrainerConfig(
        method_name="r3f",
        steps_per_eval_batch=1000,
        steps_per_eval_image=5000,
        steps_per_save=5000,
        max_num_iterations=25000,
        mixed_precision=True,
        log_gradients=False,
        pipeline=R3FPipelineConfig(
            datamanager=RefRefDataManagerConfig(
                dataparser=RefRefDataParserConfig,
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=8192,
            ),
            model=R3FModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                gin_file=["configs/refref.gin"],
                proposal_weights_anneal_max_num_iters=1000,
            ),
        ),
        optimizers={
            "model": {
                "optimizer": AdamOptimizerConfig(lr=8e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(warmup_steps=1000,lr_final=1e-3, max_steps=25000)
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="A PyTprch implementation of RefRef: A Synthetic Dataset and Benchmark for Reconstructing Refractive and Reflective Objects (https://arxiv.org/abs/2505.05848). ",
)
