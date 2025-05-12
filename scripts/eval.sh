#!/usr/bin/env bash
#SBATCH --job-name=eval_est
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --mem=24gb
#SBATCH --gres=gpu:4090:1
#SBATCH --account=optimalvision
#SBATCH --qos=gpu-b
#SBATCH --partition=gpu
##SBATCH --partition=teaching
#SBATCH --output=logs_eval/%j.log
#SBATCH --time=5-00:00:00

pwd; hostname; date
singularity shell --nv /opt/apps/containers/conda/conda-nvidia-22.04-latest.sif
PREFIX=nerfstudio_2080ti
CONDA_INIT_SCRIPT=/home/projects/u7535192/anaconda3/etc/profile.d/conda.sh
source $CONDA_INIT_SCRIPT
conda activate $PREFIX
echo "Training started."
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'


#CATEGORY_NAME=simple_shapes
#CATEGORY_NAME=complex_shapes
#CATEGORY_NAME=household_items
#CATEGORY_NAME=lab_equipment
#
##ckpt_dir=/home/projects/u7535192/projects/zipnerf-pytorch/outputs_hdr/zip_oracle_complex_shapes_cat/zipnerf/2025-01-30_114002
##ns-eval --load-config $ckpt_dir/config.yml \
##                    --output-path $ckpt_dir/output_test.json \
##                    --render-output-path $ckpt_dir/output_images
#
##for data_dir in /home/projects/RefRef/image_data/HDR_map_scene/$CATEGORY_NAME/*; do
##for data_dir in $(ls -d /home/projects/RefRef/image_data/HDR_map_scene/$CATEGORY_NAME/* | tac); do
#for data_dir in /home/projects/RefRef/image_data/textured_cube_scene/$CATEGORY_NAME/*; do
##for data_dir in $(ls -d /home/projects/RefRef/image_data/textured_sphere_scene/$CATEGORY_NAME/* | tac); do
#    category_dir=$(basename "$(dirname "$data_dir")")  # Get the parent directory of data_dir
#    current_dir=$(basename "$data_dir")  # Get the current directory
#    echo $current_dir
#    dataset_name="${category_dir}_${current_dir}"
#
#        ckpt_dir=$(find outputs_new_mesh/zip_oracle_"${dataset_name}" -type d -name "2025-02-17_014553")
#        ckpt_dir=/home/projects/u7543832/outputs/neus_mask_simple_shapes_ball_mask/neus-facto/2025-02-18_120823
#        echo "Found checkpoint: $ckpt_dir"
#        if [ -f "$ckpt_dir/output_test.json" ]; then
#              echo "Output already exists. Skipping..."
#              continue
#        fi
#
#        run_dirs=(
#          "ball"
#        )
#
#        if [[ " ${run_dirs[@]} " =~ " $current_dir " ]]; then
#            echo "Running $current_dir"
#
#          if [ -f "$ckpt_dir/nerfstudio_models/step-000025000.ckpt" ]; then
#              ns-eval --load-config $ckpt_dir/config.yml \
#                      --output-path $ckpt_dir/output_test.json \
#                      --render-output-path $ckpt_dir/output_images
#              ns-export pointcloud --load-config $ckpt_dir/config.yml --output-dir mesh_files/
#              ns-extract-mesh --load-config $ckpt_dir/config.yml --output-path mesh_files/neus-facto-ball_mask.ply
#          fi
#        fi
#done

#python depth2pcd.py --depth_file /home/projects/u7535192/projects/zipnerf-pytorch/outputs_new_mesh/zip_oracle_simple_shapes_ball/zipnerf/2025-02-19_122433/output_images/depth_maps_npy/r_0_depth_gt.npy \
#--fx 610.0 --fy 610.0 --cx 400 --cy 400 --output mesh_files/pcd_gt_0.ply
#python depth2pcd.py --depth_file /home/projects/u7535192/projects/zipnerf-pytorch/outputs_new_mesh/zip_oracle_simple_shapes_ball/zipnerf/2025-02-19_122433/output_images/depth_maps_npy/r_0_depth_pred.npy \
#--fx 610.0 --fy 610.0 --cx 400 --cy 400 --output mesh_files/pcd_pred_0.ply
#python depth2pcd.py --depth_file /home/projects/u7535192/projects/Depth-Anything-V2/metric_depth/test_outputs/ball/train/r_0_raw_depth_meter.npy \
#--fx 610.0 --fy 610.0 --cx 400 --cy 400 --output mesh_files/pcd_ball_0.ply

#chmod -R 777 outputs_zipnerf_oracle/

outputs_dir=outputs_zipnerf_oracle_est_geometry
#category=single-convex
#category=single-non-convex
#category=multiple-non-convex
for result_dir in $outputs_dir/*multiple*teacup*; do
#for result_dir in $(ls -d $outputs_dir/*multiple* | tac); do
#for result_dir in $outputs_dir/*${category}*; do
#for result_dir in $(ls -d $outputs_dir/*${category}* | tac); do
#if [[ "$result_dir" == *hdr* || "$result_dir" == *sphere* ]]; then
#    continue  # Skip folders containing "hdr" or "sphere"
#fi
    echo "Processing" $result_dir

    ckpt_dir=$(find $result_dir -type d -name "2025-*")
    echo "Found checkpoint directory: $ckpt_dir"

    if [ -f "$ckpt_dir/output.json" ]; then
              echo "Output already exists. Skipping..."
              continue
    fi

    # skip if result_dir ends with "hdr"
#    if [[ "$result_dir" == *sphere* ]]; then
#        continue
#    fi

    if [ -f "$ckpt_dir/nerfstudio_models/step-000025000.ckpt" ]; then
        ns-eval --load-config $ckpt_dir/config.yml \
                --output-path $ckpt_dir/output.json \
                --render-output-path $ckpt_dir/output_images
    fi
#    fi
done


#python get_outputs.py ./outputs_zipnerf_oracle_est_geometry pythia_dist.xlsx
#python get_outputs.py /home/projects/u7543832/eval_splat outputs_splat.xlsx
#python get_outputs.py /home/projects/u7543832/eval_result_splat outputs_splat.xlsx