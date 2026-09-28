#################################### Sam 3 Segmentation ####################################
BG=living_room
BG=puzzle_table
BG=courtyard
BG=lookout_deck

python utils/sam3_infer.py real_captures/$BG/bulb/images \
    --prompt "bottle" \
    --output_path real_captures/$BG/bulb/masks

python utils/sam3_infer.py real_captures/$BG/dishwashing_liquid/images \
    --prompt "bottle" \
    --output_path real_captures/$BG/dishwashing_liquid/masks

python utils/sam3_infer.py real_captures/$BG/glass_of_water/images \
    --prompt "glass of water" \
    --output_path real_captures/$BG/glass_of_water/masks

python utils/sam3_infer.py real_captures/$BG/glass_with_straw/images \
    --prompt "glass" \
    --output_path real_captures/$BG/glass_with_straw/masks

python utils/sam3_infer.py real_captures/$BG/measuring_jug/images \
    --prompt "glass" \
    --output_path real_captures/$BG/measuring_jug/masks

python utils/sam3_infer.py real_captures/$BG/perfume/images \
    --prompt "bottle" \
    --output_path real_captures/$BG/perfume/masks

python utils/sam3_infer.py real_captures/$BG/perfume_set/images \
    --prompt "perfume bottle" \
    --output_path real_captures/$BG/perfume_set/masks

python utils/sam3_infer.py real_captures/$BG/pyramid/images \
    --prompt "glass pyramid" \
    --output_path real_captures/$BG/pyramid/masks

python utils/sam3_infer.py real_captures/$BG/rabbit/images \
    --prompt "glass object" \
    --output_path real_captures/$BG/rabbit/masks

python utils/sam3_infer.py real_captures/$BG/cocktail_shaker/images \
    --prompt "bottle" \
    --output_path real_captures/$BG/cocktail_shaker/masks

python utils/sam3_infer.py real_captures/$BG/snow_globe/images \
    --prompt "snow globe" \
    --output_path real_captures/$BG/snow_globe/masks

python utils/sam3_infer.py real_captures/$BG/snow_globe1/images \
    --prompt "crystal ball" \
    --output_path real_captures/$BG/snow_globe1/masks

python utils/sam3_infer.py real_captures/$BG/sphere/images \
    --prompt "glass sphere" \
    --output_path real_captures/$BG/sphere/masks

python utils/sam3_infer.py real_captures/$BG/wine_bottle_set/images \
    --prompt "glass object" \
    --output_path real_captures/$BG/wine_bottle_set/masks

python utils/sam3_infer.py real_captures/$BG/wine_bottle/images \
    --prompt "bottle" \
    --output_path real_captures/$BG/wine_bottle/masks

bash scripts/ns_transforms.sh real_captures/lookout_deck/
python scripts/inject_splits.py real_captures/lookout_deck/*/

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
BG=textured_sphere_scene
for shape_folder in /home/projects/RefRef/image_data/$BG/m*/; do
    shape_type=$(basename "$shape_folder")
    echo "Processing: $shape_type"
    # for scene_folder in $shape_folder/*/; do
    for scene_folder in $(ls -d $shape_folder/*/ | sort -r); do
        dataset_name=$(basename "$scene_folder")
        dataset_path="$scene_folder"

        # Skip if ckpt step-000009999.ckpt exits
        if ls outputs/fg/r3f_${dataset_name}_fg/r3f/*/nerfstudio_models/*.ckpt 2>/dev/null | grep -q .; then
            echo "Skipping $dataset_name: checkpoint already exists"
            continue
        fi
        
        bg_ckpt=$(find outputs/bg/r3f_${dataset_name}/r3f/*/ -type f -name "*.ckpt" | head -n 1)
        ply_file=$(find outputs/meshes/ -type f -name "${dataset_name%_sphere}_*.ply" | sort | tail -n 1)
        if [ -z "$ply_file" ]; then
            echo "Skipping $dataset_name: no matching mesh file found"
            continue
        fi

        WANDB_TMPDIR=$(mktemp -d)
        export WANDB_DIR="$WANDB_TMPDIR"

        ns-train r3f --pipeline.stage fg \
                    --pipeline.bg-checkpoint-path "$bg_ckpt" \
                    --machine.device-type cuda \
                    --machine.num-devices 1 \
                    --project-name r3f \
                    --experiment-name "r3f_${dataset_name}_fg" \
                    --pipeline.datamanager.train-num-workers 2 \
                    --pipeline.datamanager.eval-num-workers 2 \
                    --pipeline.bg-far 1000 \
                    --pipeline.bg-opaque-background True \
                    --pipeline.model.gin-file "configs/refref_fg.gin" \
                    --pipeline.model.background-color random \
                    --max-num-iterations 10000 \
                    --steps_per_eval_image 1000 \
                    --vis wandb \
                    --data "$dataset_path" \
                    --output-dir "outputs/fg" \
                blender-refref-data \
                    --scale-factor 0.1 \
                    --ply-path "$ply_file"
    
    done
    rm -rf "$WANDB_TMPDIR"
    rm -rf outputs/fg/r3f_*/r3f/*/wandb
done

#################################### Evaluate #####################################
# BG_list=(env_map_scene textured_sphere_scene textured_cube_scene)
BG_list=(textured_sphere_scene)
for BG in "${BG_list[@]}"; do
    for shape_folder in /home/projects/RefRef/image_data/$BG/*/; do
        shape_type=$(basename "$shape_folder")
        echo "Processing: $shape_type"
        for scene_folder in $shape_folder/*/; do
        # for scene_folder in $(ls -d $shape_folder/*/ | sort -r); do
            dataset_name=$(basename "$scene_folder")
            dataset_path="$scene_folder"

            # skip if scene_folder name is not in the list below
            list_of_scenes=("beaker_sphere" "cube_sphere" "cylinder_coloured_sphere" "reed_diffuser_sphere")
            if [[ ! " ${list_of_scenes[@]} " =~ " ${dataset_name} " ]]; then
                continue
            fi

            # Search every run folder and take the latest step-000009999.ckpt if it exists.
            ckpt=$(find "outputs/fg/r3f_${dataset_name}_fg/r3f" -type f -name "step-000009999.ckpt" | sort -r | head -n 1)
            if [ -z "$ckpt" ]; then
                echo "Skipping $dataset_name: checkpoint does not exist"
                continue
            fi

            # Skip if the last evaluation output (r_99.png) already exists (e.g., outputs_r3f/${dataset_name}/rgb_images/r_99.png)
            if [ -f "outputs_r3f_vis/${dataset_name}/rgb_images/r_99.png" ]; then
                echo "Skipping $dataset_name: evaluation already exists"
                continue
            fi

            ckpt_dir=$(dirname "$(dirname "$ckpt")")
            ns-eval --load-config "$ckpt_dir/config.yml" \
                    --output-path "$ckpt_dir/output.json" \
                    --render-output-path "outputs_r3f_vis/${dataset_name}"
        done
    done
done

################################## Train and evaluate on 1 scene  ##################################
# BG stage
dataset_name="ball_real"
dataset_path="/home/projects/u7535192/projects/RefRef/real/ball_real"
WANDB_TMPDIR=$(mktemp -d)
export WANDB_DIR="$WANDB_TMPDIR"

ns-train r3f --pipeline.stage bg \
                    --machine.device-type cuda \
                    --machine.num-devices 1 \
                    --project-name r3f \
                    --experiment-name "r3f_${dataset_name}" \
                    --pipeline.model.gin-file "configs/refref.gin" \
                    --pipeline.model.background-color random \
                    --max-num-iterations 10000 \
                    --steps_per_eval_image 1000 \
                    --vis wandb \
                    --data "$dataset_path" \
                    --output-dir "outputs/real/bg" \
                blender-refref-data \
                    --scale-factor 0.1
rm -rf "$WANDB_TMPDIR"
rm -rf outputs/real/bg/r3f_*/r3f/*/wandb

# FG stage
bg_ckpt="/home/projects/u7535192/projects/refref/outputs/real/bg/r3f_ball_real/r3f/2026-03-12_200733"
ply_file="/home/projects/u7535192/projects/refref/ball_real_glass.ply"
ns-train r3f --pipeline.stage fg \
            --pipeline.bg-checkpoint-path "$bg_ckpt" \
            --machine.device-type cuda \
            --machine.num-devices 1 \
            --project-name r3f \
            --experiment-name "r3f_${dataset_name}_fg" \
            --pipeline.datamanager.train-num-workers 2 \
            --pipeline.datamanager.eval-num-workers 2 \
            --pipeline.bg-far 1000 \
            --pipeline.bg-opaque-background True \
            --pipeline.model.gin-file "configs/refref_fg.gin" \
            --pipeline.model.background-color random \
            --max-num-iterations 10000 \
            --steps_per_eval_image 1000 \
            --vis wandb \
            --data "$dataset_path" \
            --output-dir "outputs/real/fg" \
        blender-refref-data \
            --scale-factor 0.1 \
            --ply-path "$ply_file"




ckpt_dir=/home/projects/u7535192/projects/RefRef_results/outputs_oracle/oracle_single-convex_pyramid_hdr/r3f/2025-06-18_014103
ns-eval --load-config $ckpt_dir/config.yml \
        --output-path $ckpt_dir/output.json \
        --render-output-path $ckpt_dir/outputs

################################# Evaluate Oracle ###################################
ckpt_dir=/home/projects/u7535192/projects/RefRef_results/outputs_oracle/oracle_multiple-non-convex_beaker/zipnerf/2025-02-21_042023/
ns-eval --load-config $ckpt_dir/config.yml \
        --output-path $ckpt_dir/output.json \
        --render-output-path "outputs_oracle_vis/beaker"
