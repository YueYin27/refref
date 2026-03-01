#################################### Stage 1: BG (background) ####################################
BG=env_map_scene
for shape_folder in /home/projects/RefRef/image_data/$BG/*/; do
    shape_type=$(basename "$shape_folder")
    echo "Processing: $shape_type"
    for scene_folder in $shape_folder/*/; do
        dataset_name=$(basename "$scene_folder")
        dataset_path="$scene_folder"

        # Skip if ckpt step-000009999.ckpt exits (e.g., /home/projects/u7535192/projects/refref/outputs/bg/r3f_ampoule_hdr/r3f/2026-02-28_041915/nerfstudio_models/step-000009999.ckpt)
        if ls outputs/bg/r3f_${dataset_name}/r3f/*/nerfstudio_models/step-000009999.ckpt 2>/dev/null | grep -q .; then
            echo "Skipping $dataset_name: checkpoint already exists"
            continue
        fi

        echo "Training on dataset: $dataset_name"
        WANDB_TMPDIR=$(mktemp -d)
        export WANDB_DIR="$WANDB_TMPDIR"

        ns-train r3f --pipeline.stage bg \
                    --machine.device-type cuda \
                    --machine.num-devices 1 \
                    --project-name r3f \
                    --experiment-name "r3f_${dataset_name}" \
                    --pipeline.model.gin-file "configs/refref_hdr.gin" \
                    --pipeline.model.background-color random \
                    --max-num-iterations 10000 \
                    --steps_per_eval_image 1000 \
                    --vis wandb \
                    --data "$dataset_path" \
                    --output-dir "outputs/bg" \
                blender-refref-data \
                    --scale-factor 0.1

    rm -rf "$WANDB_TMPDIR"
    rm -rf outputs/bg/r3f_*/r3f/*/wandb
    done
done

#################################### Stage 2: FG (foreground) ####################################
dataset_name="ball"
dataset_path="/workspace/image_data/textured_cube_scene/single-convex/${dataset_name}"
bg_ckpt=$(find outputs/r3f_${dataset_name}/r3f/*/ -type f -name "*.ckpt" | head -n 1)
ply_files=$(find outputs/meshes/ -type f -name "${dataset_name%_*}*.ply" | tr '\n' ' ')
WANDB_TMPDIR=$(mktemp -d)
export WANDB_DIR="$WANDB_TMPDIR"

ns-train r3f --pipeline.stage fg \
            --pipeline.bg-checkpoint-path "$bg_ckpt" \
            --machine.device-type cuda \
            --machine.num-devices 1 \
            --project-name r3f \
            --experiment-name "r3f_${dataset_name}_fg" \
            --pipeline.model.gin-file "configs/refref_fg.gin" \
            --pipeline.model.background-color random \
            --max-num-iterations 20000 \
            --steps_per_eval_image 100 \
            --vis wandb \
            --data "$dataset_path" \
            --output-dir "outputs/fg" \
        blender-refref-data \
            --scale-factor 0.1 \
            --ply-path "${ply_files}"

rm -rf "$WANDB_TMPDIR"
rm -rf outputs/fg/r3f_*/r3f/*/wandb