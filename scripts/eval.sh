#!/usr/bin/env bash
#SBATCH --job-name=eval
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
PREFIX=nerfstudio-refref
CONDA_INIT_SCRIPT=/home/projects/u7535192/anaconda3/etc/profile.d/conda.sh
source $CONDA_INIT_SCRIPT
conda activate $PREFIX
echo "Training started."
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export HF_CACHE_DIR="/home/projects/u7535192/.cache/huggingface"

cd..
ckpt_dir=outputs/r3f_oracle_cube/r3f/2025-05-14_154414
ns-eval --load-config $ckpt_dir/config.yml \
                --output-path $ckpt_dir/output.json \
                --render-output-path $ckpt_dir/output_images
