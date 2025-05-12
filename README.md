<h1 align="center" style="font-size: 36px; margin-bottom: 10px;">RefRef: A Synthetic Dataset and Benchmark for Reconstructing Refractive and Reflective Objects</h1>

<div align="center" style="margin-bottom: 20px;">
  <a href="">Yue Yin</a> · 
  <a href="">Enze Tao</a> · 
  <a href="https://weijiandeng.xyz/">Weijian Deng</a> · 
  <a href="https://sites.google.com/view/djcampbell">Dylan Campbell</a>
</div>


<p align="center">
  <a href="https://arxiv.org/abs/2505.05848">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoColor=white" style="height: 27px; margin: 5px;">
  </a>
  <a href="https://huggingface.co/datasets/yinyue27/RefRef">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?logo=huggingface&logoColor=white" style="height: 27px; margin: 5px;">
  </a>
  <img src="https://img.shields.io/badge/Project-Website-blue?logo=google-chrome&logoColor=white" style="height: 27px; margin: 5px;">
</p>

<br>

---



## ✨ Overview
This repository provides both a synthetic dataset and a novel method for reconstructing scenes with refractive and reflective objects from posed images:

- **RefRef Dataset**: 150 high-quality synthetic scenes containing reflective and refractive objects;  
- **Oracle Method**: a method that models light paths using ground-truth object geometry and refractive indices;
- **R3F (Refractive–Reflective Radiance Field)**: a method that relaxes these requirements by estimating and smoothing the object geometry.  

## 🚀 Quick Start

### 🛠️ Setup the Environment

1. **Install [nerfstudio](https://github.com/nerfstudio-project/nerfstudio):**

      You can simply install nerfstudio using pip:
      ```bash
      pip install nerfstudio
      ```
      or install it from source, detailed instructions can be found [here](https://github.com/nerfstudio-project/nerfstudio#1-installation-setup-the-environment). Here, we also provide a quick installation script for your convenience:
      ```bash
      conda create --name nerfstudio -y python=3.8
      conda activate nerfstudio
        
      pip install --upgrade pip
      pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
      conda install -y -c "nvidia/label/cuda-11.7.1" cuda-toolkit
      pip install setuptools==69.5.1
      pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
        
      git clone https://github.com/nerfstudio-project/nerfstudio.git
      cd nerfstudio
      pip install --upgrade pip setuptools
      pip install -e .
      ```

2. **Clone this repository and install dependencies:**
      ```bash
      git clone https://github.com/YueYin27/r3f-refref.git
      cd RefRef
      pip install -e .
      pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
      ```

3. **Verify the install by running the help command for our training script:**
      ```bash
      ns-train r3f refref-data --help
      ```

### 🗂️ Accessing the RefRef Dataset
  The RefRef dataset is hosted on Hugging Face and designed for seamless integration with the `refref_dataparser`.
1. **Scene List**: All available scenes are listed here: [scene_list.txt](https://huggingface.co/datasets/yinyue27/RefRef/blob/main/scene_list.txt)
2. **Automatic Loading**: Images and camera poses auto-downloaded when you run `ns-train` or `ns-eval`. Just specify the `scene-name` from the list above.

3. **Manual Step (Required)**:  
You must download the corresponding `.ply` mesh file locally:  
    ```bash
    # Example: Download cube mesh
    wget https://huggingface.co/datasets/yinyue27/RefRef/resolve/main/mesh_files/single-material_convex/cube.ply -O ./mesh_files/cube.ply
    ```

### 🔄 Example for Optimizing a Scene
Here is a sample training command (replace placeholders as needed):
```
ply_file=./mesh_files/single-material_convex/cube.ply  # Path to your .ply file
ns-train r3f --machine.device-type cuda \
             --machine.num-devices 1 \
             --project-name r3f-refref \
             --experiment-name "r3f_cube" \
             --pipeline.model.gin-file "configs/refref.gin" \
             --pipeline.model.background-color random \
             --max-num-iterations 25001 \
             --steps-per-eval-all-images 26000 \
             --steps_per_eval_image 1000 \
             --vis wandb \
             --output-dir "outputs" \
         refref-data \
             --scene-name "cube_smcvx_cube" \
             --scale-factor 0.1 \
             --ply-path "$ply_file" \
```

### 📊 Example for Evaluating an Optimized Scene
```
# Path to your output checkpoint folder
ckpt_dir=./outputs/r3f-cube/r3f/2025-05-09_201637
ns-eval --load-config $ckpt_dir/config.yml \
                --output-path $ckpt_dir/output.json \
                --render-output-path $ckpt_dir/output_images
```

## 📑 Citation  
```
@misc{yin2025refrefsyntheticdatasetbenchmark,
      title={RefRef: A Synthetic Dataset and Benchmark for Reconstructing Refractive and Reflective Objects}, 
      author={Yue Yin and Enze Tao and Weijian Deng and Dylan Campbell},
      year={2025},
      eprint={2505.05848},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.05848}, 
}
```

## 🙏 Acknowledgements
Our project is built on the [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) framework, and [an pytorch implementation of ZipNeRF](https://github.com/SuLvXiangXin/zipnerf-pytorch/tree/main).
Thanks to [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) for their great work on the nerfstudio framework, and [ZipNeRF](https://github.com/SuLvXiangXin/zipnerf-pytorch/tree/main) for their great implementation of ZipNeRF.


## ⚖️ License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.