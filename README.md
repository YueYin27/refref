<h1 align="center" style="font-size: 36px; margin-bottom: 10px;">RefRef: A Synthetic Dataset and Benchmark for Reconstructing Refractive and Reflective Objects</h1>

<div align="center" style="margin-bottom: 20px;">
  <a href="">Yue Yin</a> · 
  <a href="">Enze Tao</a> · 
  <a href="https://weijiandeng.xyz/">Weijian Deng</a> · 
  <a href="https://sites.google.com/view/djcampbell">Dylan Campbell</a>
</div>

<br>

<p align="center">
  <a href="https://arxiv.org/abs/2505.05848">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoColor=white" style="height: 27px; margin: 5px;">
  </a>&nbsp;
  <a href="https://huggingface.co/datasets/yinyue27/RefRef">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface&logoColor=white" style="height: 27px; margin: 5px;">
  </a>&nbsp;
  <img src="https://img.shields.io/badge/Project-Website-blue?logo=google-chrome&logoColor=white" style="height: 27px; margin: 5px;">
</p>

<br>

## ✨ Overview
This repository provides both a synthetic dataset and a novel method for reconstructing scenes with refractive and reflective objects from posed images:

- **RefRef Dataset**: 150 high-quality synthetic scenes containing reflective and refractive objects;  
- **Oracle Method**: a method that models light paths using ground-truth object geometry and refractive indices;
- **R3F (Refractive–Reflective Radiance Field)**: a method that relaxes these requirements by estimating and smoothing the object geometry.  

<br>

## 🚀 Quickstart

### 🛠️ Setup the Environment

1. **Create a conda environment and install dependencies:**
   ```bash
   conda create --name r3f -y python=3.8
   conda activate r3f
   
   pip install --upgrade pip
   pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
   conda install -y -c "nvidia/label/cuda-11.7.1" cuda-toolkit
   pip install setuptools==69.5.1
   pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
   ```
2. **Install [nerfstudio](https://github.com/nerfstudio-project/nerfstudio):**

      You can simply install nerfstudio using pip:
      ```bash
      pip install nerfstudio
      ```
      or install it from source:
      ```bash
      git clone https://github.com/nerfstudio-project/nerfstudio.git
      cd nerfstudio
      pip install --upgrade pip setuptools
      pip install -e .
      ```

3. **Install [sdfstudio](https://github.com/autonomousvision/sdfstudio):**
    ```bash
    git clone https://github.com/autonomousvision/sdfstudio.git
    cd sdfstudio
    pip install --upgrade pip setuptools
    pip install -e .
    ```

4. **Clone this repository and install dependencies:**
      ```bash
      git clone https://github.com/YueYin27/refref.git
      cd refref
   
      pip install -r requirements.txt
      pip install ./extensions/cuda
      pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
      pip install -e .
      ns-install-cli
      ```

5. **Verify the installation by running the help command for our training script:**
      ```bash
      ns-train r3f --help
      ns-train r3f refref-data --help
      ```

### 🗂️ Accessing the [RefRef Dataset](https://huggingface.co/datasets/yinyue27/RefRef)
  The RefRef dataset is hosted on Hugging Face and designed for seamless integration with the `refref_dataparser`.
1. **Scene List**: All available scenes are listed here: [scene_list.txt](https://huggingface.co/datasets/yinyue27/RefRef/blob/main/scene_list.txt)
2. **Automatic Loading**: Images and camera poses auto-downloaded when you run `ns-train` or `ns-eval`. Just specify the `scene-name` from the list above.


### 🔄 Optimize a Scene

1. **Stage-1: Estimate and smooth the object geometry**
   - <details>
     <summary>Download masks from <a href="https://huggingface.co/datasets/yinyue27/RefRef_additional">this HuggingFace repository</a></summary>
     
     ```bash
     wget https://huggingface.co/datasets/yinyue27/RefRef_additional/blob/main/masks.zip -O ./masks.zip
     unzip masks.zip
     ```
     </details>
   - <details>
     <summary>Train <a href="https://github.com/autonomousvision/unisurf">UNISURF</a> (we provide a sample scripts)</summary>

      ```bash
      data_dir=./data/.../cube  # Path to mask images
      ns-train unisurf --trainer.max-num-iterations 20000 \
                       --pipeline.model.sdf-field.inside-outside False \
                       --pipeline.model.near-plane 0.05 \
                       --pipeline.model.far-plane 6 \
                       --pipeline.model.far-plane-bg 20 \
                       --pipeline.model.sdf-field.bias 0.8 \
                       --pipeline.model.sdf-field.beta-init 0.5 \
                       --pipeline.model.sdf-field.use_diffuse_color True \
                       --pipeline.model.sdf-field.use-grid-feature True \
                       --pipeline.model.smooth-loss-multi 0.05 \
                       --pipeline.model.background-model none \
                       --pipeline.model.background-color black \
                       --pipeline.model.overwrite_near_far_plane True \
                       --optimizers.fields.optimizer.lr 0.0001 \
                       --vis wandb \
               blender-data \
                       --scale-factor 0.5 \
                       --data "$data_dir"
      ```
      </details>
   - <details>
     <summary>Export the mesh file</summary>
     
     ```bash
     config_path=./outputs/.../config.yml  # Path to your output config file
     output_path=./outputs/.../cube.ply  # Path to your output mesh
     ns-extract-mesh --load-config $config_path \
                     --output-path $output_path \
                     --bounding_box_min -2.1 -2.1 -2.1 \
                     --bounding_box_max 2.1 2.1 2.1 \
                     --is_occupancy True \
                     --resolution 700 \
                     --simplify_mesh True \
                     --torch_precision highest
      ```
     </details>

   - Smooth the mesh file following the instructions [here](https://arxiv.org/pdf/2505.05848).

2. **Stage 2: Optimize a Scene**
    - **Run R3F:**
      ```bash
      ply_file=./mesh_files/.../cube_est.ply  # Path to your estimated mesh file
      ns-train r3f --machine.device-type cuda \
                   --machine.num-devices 1 \
                   --project-name r3f \
                   --pipeline.model.gin-file "configs/refref.gin" \
                   --pipeline.model.background-color random \
                   --max-num-iterations 25000 \
                   --output-dir "outputs" \
               refref-data \
                   --scene-name "cube_smcvx_cube" \
                   --scale-factor 0.1 \
                   --ply-path "$ply_file"
      ```
    - **Run Oracle:**
      ```bash
      # Download the ground truth mesh files
      wget https://huggingface.co/datasets/yinyue27/RefRef_additional/blob/main/mesh_files.zip -O ./masks.zip
      unzip mesh_files.zip
      
      # Run Oracle
      ply_file=./mesh_files/.../cube_gt.ply  # Path to your ground truth mesh file
      ns-train r3f --machine.device-type cuda \
                   --machine.num-devices 1 \
                   --project-name oracle \
                   --pipeline.model.gin-file "configs/refref.gin" \
                   --pipeline.model.background-color random \
                   --max-num-iterations 25000 \
                   --output-dir "outputs" \
               refref-data \
                   --scene-name "cube_smcvx_cube" \
                   --scale-factor 0.1 \
                   --ply-path "$ply_file"
      ```

### 📊 Evaluate an Optimized Scene
   ```bash
   # Path to your output checkpoint folder
   config_path=./outputs/.../config.yml  # Path to your output config file
   output_path=./outputs/.../output.json  # Path to your output json file
   output_img_dir=./outputs/.../output_images  # Path to your output image folder
   ns-eval --load-config $config_path \
           --output-path $output_path \
           --render-output-path $output_img_dir
   ```

<br>

## 📑 Citation  
   ```bibtex
   @misc{yin2025refrefsyntheticdatasetbenchmark,
         title={RefRef: A Synthetic Dataset and Benchmark for Reconstructing Refractive and Reflective Objects}, 
         author={Yue Yin and Enze Tao and Weijian Deng and Dylan Campbell},
         year={2025},
         eprint={2505.05848},
         archivePrefix={arXiv},
         primaryClass={cs.CV},
         url={https://arxiv.org/abs/2505.05848}, 
   }
   
   @misc{barron2023zipnerf,
      title={Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields}, 
      author={Jonathan T. Barron and Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman},
      year={2023},
      eprint={2304.06706},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
   }
   
   @inproceedings{nerfstudio,
	title        = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
	author       = {
		Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi, Brent
		and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
		Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa,
		Angjoo
	},
	year         = 2023,
	booktitle    = {ACM SIGGRAPH 2023 Conference Proceedings},
	series       = {SIGGRAPH '23}
    }
    
    @misc{Yu2022SDFStudio,
    author    = {Yu, Zehao and Chen, Anpei and Antic, Bozidar and Peng, Songyou and Bhattacharyya, Apratim 
                 and Niemeyer, Michael and Tang, Siyu and Sattler, Torsten and Geiger, Andreas},
    title     = {SDFStudio: A Unified Framework for Surface Reconstruction},
    year      = {2022},
    url       = {https://github.com/autonomousvision/sdfstudio},
    }
   ```

<br>

## 🙏 Acknowledgements
Our project is built on the [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) framework, and [an pytorch implementation of ZipNeRF](https://github.com/SuLvXiangXin/zipnerf-pytorch/tree/main).
Thanks to [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) for their great work on the nerfstudio framework, and [ZipNeRF](https://github.com/SuLvXiangXin/zipnerf-pytorch/tree/main) for their great implementation of ZipNeRF.

<br>

## ⚖️ License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
