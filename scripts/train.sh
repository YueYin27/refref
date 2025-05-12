#!/usr/bin/env bash
#SBATCH --job-name=zip_lab
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --mem=24gb
#SBATCH --gres=gpu:4090:1
#SBATCH --account=optimalvision
#SBATCH --qos=gpu-b
#SBATCH --partition=gpu
##SBATCH --partition=teaching
#SBATCH --output=logs/%j.log
#SBATCH --time=5-00:00:00

pwd; hostname; date
singularity shell --nv /opt/apps/containers/conda/conda-nvidia-22.04-latest.sif
PREFIX=nerfstudio_2080ti
CONDA_INIT_SCRIPT=/home/projects/u7535192/anaconda3/etc/profile.d/conda.sh
source $CONDA_INIT_SCRIPT
conda activate $PREFIX
echo "Training started."
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

export WANDB_API_KEY=bae68d373174c2076d0d0a9e390157a84a72bf04
wandb login
export HF_CACHE_DIR="/home/projects/u7535192/.cache/huggingface"

CATEGORY_NAME=single-convex
#CATEGORY_NAME=single-non-convex
#CATEGORY_NAME=multiple-non-convex
for data_dir in /home/projects/RefRef/image_data/textured_cube_scene/$CATEGORY_NAME/*; do
#for data_dir in /home/projects/RefRef/image_data/textured_sphere_scene/$CATEGORY_NAME/*; do
#for data_dir in /home/projects/RefRef/image_data/HDR_map_scene/$CATEGORY_NAME/*; do
#for data_dir in $(ls -d /home/projects/RefRef/image_data/HDR_map_scene/$CATEGORY_NAME/* | tac); do
    category_dir=$(basename "$(dirname "$data_dir")")  # e.g. simple_shapes
    current_dir=$(basename "$data_dir")  # e.g. ball
    dataset_name="${category_dir}_${current_dir}"  # e.g. simple_shapes_ball
    echo "Processing $dataset_name"

    if [[ "$current_dir" =~ (_sphere|_hdr)$ ]]; then
      current_dir="${current_dir%${BASH_REMATCH[1]}}"
    fi

    # get the ply file path
    ply_files=""
    for material in glass water alcohol perfume plastic diamond; do
        file="/home/projects/RefRef/mesh_files/$CATEGORY_NAME/${current_dir}_${material}.ply"
#        file="/home/projects/RefRef/mesh_files/$CATEGORY_NAME/${current_dir}_${material}_est.ply"
        if [ -f "$file" ]; then
            ply_files="$ply_files $file"
        fi
    done

    echo "Found ply files: $ply_files"

#    run_dirs1=(
#          "plastic_bottle"
#          "reed_diffuser"
#          )

    run_dirs1=(
          "cube"
          )

    # Check if the current directory is in the list of directories to run
    if [[ " ${run_dirs1[@]} " =~ " $current_dir " ]]; then
        echo "Running $current_dir"

#    skip_dirs=(
#    "ball"
#    "ball_coloured"
#    )
#    ## Check if the current directory is in the list of directories to skip
#     if [[ " ${skip_dirs[@]} " =~ " $current_dir " ]]; then
#         echo "Skipping $current_dir"
#         continue
#     fi

#    CKPT_PATH=$(find outputs_zipnerf_oracle_est_geometry/zipnerf_oracle_"${dataset_name}" -type d -name "2025-*")
#    echo "Found checkpoint: $CKPT_PATH"
#    if [ -f "$CKPT_PATH/nerfstudio_models/step-000025000.ckpt" ]; then
#          echo "ckpt already exists. Skipping..."
#          continue
#    fi
    # check all packages and their versions
    pip list
    conda list
    pip install -e .
#    ns-train r3f --help
#    ns-train r3f hf-data --help
#    pip install datasets

    ns-train r3f --machine.device-type cuda \
                 --machine.num-devices 1 \
                 --project-name zipnerf_oracle \
                 --experiment-name "r3f_oracle_${dataset_name}" \
                 --pipeline.model.gin-file "configs/blender.gin" \
                 --pipeline.model.background-color random \
                 --max-num-iterations 25001 \
                 --steps-per-eval-all-images 26000 \
                 --steps_per_eval_image 1000 \
                 --vis wandb \
                 --output-dir "outputs" \
             hf-data \
                 --scene-name "cube_smcvx_cube" \
                 --scale-factor 0.1 \
                 --ply-path "${ply_files}"

    fi
done
#####################
##python split_mesh.py mesh_files/vol_cylinder.ply mesh_files/vol_cylinder.ply --smooth 2
##python split_mesh.py mesh_files/unisurf_cube.ply mesh_files/unisurf_ball.ply --smooth 2
##python split_mesh.py mesh_files/vol_cylinder_coloured.ply mesh_files/cylinder_coloured.ply --smooth 2


#python video_gen.py \
#/home/projects/u7535192/projects/zipnerf-pytorch/outputs_zipnerf_oracle_est_geometry/zipnerf_oracle_single-convex_pyramid_sphere_est_geometry/zipnerf/2025-03-04_150643/output_images/rgb_images \
#/home/projects/u7535192/projects/zipnerf-pytorch/videos/ours_pyramid_sphere.mp4 --fps 30

#ns-train r3f --machine.device-type cuda \
#                 --machine.num-devices 1 \
#                 --project-name zipnerf_oracle \
#                 --experiment-name "zipnerf_oracle_${dataset_name}" \
#                 --pipeline.model.gin-file "configs/blender.gin" \
#                 --pipeline.model.background-color random \
#                 --max-num-iterations 25001 \
#                 --steps-per-eval-all-images 26000 \
#                 --steps_per_eval_image 1000 \
#                 --vis wandb \
#                 --output-dir "outputs_test" \
#             blender-data \
#                 --data "${data_dir}" \
#                 --scale-factor 0.1 \
#                 --ply-path "${ply_files}"