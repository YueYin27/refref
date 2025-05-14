#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --mem=24gb
#SBATCH --gres=gpu:a6000:1
#SBATCH --account=optimalvision
#SBATCH --qos=gpu-b
#SBATCH --partition=gpu
##SBATCH --partition=teaching
#SBATCH --output=logs/%j.log
#SBATCH --time=5-00:00:00

pwd; hostname; date
singularity shell --nv /opt/apps/containers/conda/conda-nvidia-22.04-latest.sif
CONDA_INIT_SCRIPT=/home/projects/u7535192/anaconda3/etc/profile.d/conda.sh
source $CONDA_INIT_SCRIPT
conda activate nerfstudio-refref
echo "Training started."
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export HF_CACHE_DIR="/home/projects/u7535192/.cache/huggingface"

cd ..
ns-train r3f refref-data --help
dataset_name="cube"
ply_files="/home/projects/RefRef/mesh_files/single-convex/cube_glass.ply"
ns-train r3f --machine.device-type cuda \
             --machine.num-devices 1 \
             --project-name r3f \
             --experiment-name "r3f_oracle_${dataset_name}" \
             --pipeline.model.gin-file "configs/refref.gin" \
             --pipeline.model.background-color random \
             --max-num-iterations 25000 \
             --steps_per_eval_image 1000 \
             --vis wandb \
             --output-dir "outputs" \
         refref-data \
             --scene-name "cube_smcvx_cube" \
             --scale-factor 0.1 \
             --ply-path "${ply_files}"

