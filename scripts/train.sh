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
dataset_name="ball"
ply_files="/home/projects/u7535192/projects/refref/data/real-data/glass_glass.ply"

WANDB_TMPDIR=$(mktemp -d)
export WANDB_DIR="$WANDB_TMPDIR"

ns-train r3f --pipeline.stage bg \
            --machine.device-type cuda \
            --machine.num-devices 1 \
            --project-name r3f \
            --experiment-name "r3f_${dataset_name}" \
            --pipeline.model.gin-file "configs/refref.gin" \
            --pipeline.model.background-color random \
            --max-num-iterations 25000 \
            --steps_per_eval_image 500 \
            --vis wandb \
            --data "/workspace/image_data/textured_cube_scene/single-convex/ball" \
            --output-dir "outputs" \
        blender-refref-data \
            --scale-factor 0.1

rm -rf "$WANDB_TMPDIR"
rm -rf outputs/r3f_*/r3f/*/wandb

python extract_mesh_stage1.py --cfg data/model/ball_coloured/ball_coloured.yaml

# ── Stage 2: FG (foreground in-object field) ──
# Set bg_ckpt to the checkpoint from the bg stage above
bg_ckpt="outputs/r3f_ball/r3f/2026-02-26_121600/nerfstudio_models/step-000010000.ckpt"
ply_files="outputs/r3f_ball/r3f/2026-02-26_121600/ball_glass.ply"
WANDB_TMPDIR=$(mktemp -d)
export WANDB_DIR="$WANDB_TMPDIR"

ns-train r3f --pipeline.stage fg \
            --pipeline.bg-checkpoint-path "$bg_ckpt" \
            --machine.device-type cuda \
            --machine.num-devices 1 \
            --project-name r3f \
            --experiment-name "r3f_${dataset_name}_fg" \
            --pipeline.model.gin-file "configs/refref.gin" \
            --pipeline.model.background-color random \
            --max-num-iterations 25000 \
            --steps_per_eval_image 100 \
            --vis wandb \
            --data "/workspace/image_data/textured_cube_scene/single-convex/ball" \
            --output-dir "outputs" \
        blender-refref-data \
            --scale-factor 0.1 \
            --ply-path "${ply_files}"

rm -rf "$WANDB_TMPDIR"
rm -rf outputs/r3f_*/r3f/*/wandb

