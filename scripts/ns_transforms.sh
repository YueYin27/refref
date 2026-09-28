#!/bin/bash
# Convert Metashape cameras.xml -> nerfstudio transforms.json, WITHOUT duplicating images.
#
#   bash scripts/ns_transforms.sh <dir> [<dir> ...]
#
#   <dir> may be a single scene folder (one containing cameras.xml and images/),
#   or any parent folder -- every scene beneath it is found and processed.
#
#   bash scripts/ns_transforms.sh real_captures                    # everything
#   bash scripts/ns_transforms.sh real_captures/courtyard          # one background
#   bash scripts/ns_transforms.sh real_captures/courtyard/bulb     # one scene
#   bash scripts/ns_transforms.sh --dry-run real_captures          # just list them
#
# ns-process-data always copies (and re-encodes) the images into its output dir.
# We let it do that in a temp folder, keep only transforms.json, and drop it into
# the scene folder -- where its "images/frame_XXXXX.jpg" paths already resolve
# against the original images. Re-running regenerates every scene, overwriting any
# transforms.json already there.
#
# Tool locations can be overridden from the environment, e.g.
#   R3F_BIN=/path/to/env/bin bash scripts/ns_transforms.sh real_captures

set -u

R3F_BIN="${R3F_BIN:-/home/projects/u7535192/.cache/conda/envs/r3f/bin}"
MESA_DIR="${MESA_DIR:-/home/projects/u7535192/.cache/conda/envs/r3f/nsight-compute/2022.2.1/host/linux-desktop-glibc_2_11_3-x64/Mesa}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/projects/u7535192/anaconda3/envs/nerfstudio-refref/bin}"

# cv2/open3d need libGL (absent on the host, GLIBC-mismatched inside the container);
# ffmpeg is not in r3f and runs even with --num-downscales 0. Append ffmpeg so it
# cannot shadow r3f's own ns-* binaries.
export LD_LIBRARY_PATH=$MESA_DIR:${LD_LIBRARY_PATH:-}
export PATH=$PATH:$FFMPEG_BIN

dry_run=0
if [ "${1:-}" = "--dry-run" ] || [ "${1:-}" = "-n" ]; then dry_run=1; shift; fi

if [ $# -lt 1 ]; then
    sed -n '2,16p' "$0" >&2
    exit 1
fi

is_scene() { [ -f "$1/cameras.xml" ] && [ -d "$1/images" ]; }

# Expand each argument into the list of scene folders it covers.
scenes=()
for root in "$@"; do
    root="${root%/}"
    if [ ! -d "$root" ]; then
        echo "[skip] $root - not a directory" >&2
        continue
    fi
    if is_scene "$root"; then
        scenes+=("$root")
    else
        while IFS= read -r d; do
            is_scene "$d" && scenes+=("$d")
        done < <(find "$root" -type f -name cameras.xml -printf '%h\n' | sort)
    fi
done

if [ ${#scenes[@]} -eq 0 ]; then
    echo "No scenes found (need both cameras.xml and images/) under: $*" >&2
    exit 1
fi

if [ $dry_run -eq 1 ]; then
    echo "${#scenes[@]} scene(s) would be processed:"
    for s in "${scenes[@]}"; do
        [ -f "$s/transforms.json" ] && echo "  [redo] $s" || echo "  [todo] $s"
    done
    exit 0
fi

done_=0; failed=0

for scene in "${scenes[@]}"; do
    # Always regenerate: an existing transforms.json is overwritten.
    tmp="$scene/.ns_tmp"
    rm -rf "$tmp"

    # --num-downscales 0: no images_2/4/8 (nerfstudio auto-picks scale 1 below 1600px anyway).
    if ! "$R3F_BIN/ns-process-data" metashape \
            --data "$scene/images" \
            --xml "$scene/cameras.xml" \
            --output-dir "$tmp" \
            --num-downscales 0 > "$scene/.ns_process.log" 2>&1; then
        echo "[FAIL] $scene - see $scene/.ns_process.log"
        rm -rf "$tmp"; failed=$((failed + 1)); continue
    fi

    mv "$tmp/transforms.json" "$scene/transforms.json"
    rm -rf "$tmp" "$scene/.ns_process.log"

    # ns-process-data rewrites every file_path to frame_NNNNN.jpg, numbered by sorted
    # position. Since we keep only the JSON and point it at the untouched originals,
    # any name that is not already frame_NNNNN (e.g. test_00001) would dangle. Undo it.
    if ! "$R3F_BIN/python" "$(dirname "$0")/restore_filenames.py" "$scene" > "$scene/.restore.log" 2>&1; then
        echo "[FAIL] $scene - restore_filenames.py failed, see $scene/.restore.log"
        failed=$((failed + 1)); continue
    fi
    if grep -q "FAIL" "$scene/.restore.log"; then
        echo "[FAIL] $scene - $(grep -o 'FAIL.*' "$scene/.restore.log" | head -1)"
        rm -f "$scene/.restore.log"; failed=$((failed + 1)); continue
    fi
    # The backup restore_filenames.py takes is just this run's raw ns-process-data
    # output; keeping it would leave a stale file in every scene folder.
    rm -f "$scene/.restore.log" "$scene/transforms.json.orig"

    n_img=$(ls -1 "$scene/images" | wc -l)
    n_frm=$(grep -o '"file_path"' "$scene/transforms.json" | wc -l)
    echo "[ok]   $scene - $n_frm/$n_img frames"
    [ "$n_frm" -ne "$n_img" ] && echo "       WARNING: Metashape dropped $((n_img - n_frm)) unaligned camera(s)"
    done_=$((done_ + 1))
done

echo "----"
echo "converted: $done_  failed: $failed"
