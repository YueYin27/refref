#!/usr/bin/env python
"""Rewrite transforms.json file_paths back to the original image filenames.

ns-process-data copies every input image to images/frame_NNNNN.jpg, numbered by its
position in the sorted input listing (process_data_utils.copy_images_list), and writes
those renamed paths into transforms.json. Because ns_transforms.sh keeps only the JSON
and points it at the untouched originals, any capture whose filenames are not already
frame_00001... ends up with paths that do not exist -- e.g. test_00001.jpg silently
became frame_00101.jpg.

The rename is invertible: file_path index N refers to sorted(images)[N-1]. This script
applies that inversion and cross-checks every frame against the camera labels in
cameras.xml (frames are emitted in XML order, skipping unaligned cameras). If the two
disagree the scene is left untouched and reported, rather than written wrong.

    python scripts/restore_filenames.py real_captures/*/*/
    python scripts/restore_filenames.py real_captures/park/bulb --dry-run

Idempotent: a transforms.json whose paths all resolve is left alone.
"""
import argparse
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def restore(scene_dir: Path, dry_run: bool) -> str:
    tj = scene_dir / "transforms.json"
    if not tj.is_file():
        return "skip: no transforms.json"

    meta = json.loads(tj.read_text())
    frames = meta["frames"]

    # Already correct? Then there is nothing to do.
    if all((scene_dir / f["file_path"]).exists() for f in frames):
        return f"ok (already resolved, {len(frames)} frames)"

    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        return "skip: no images/"
    originals = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS))

    # Independent source of truth: cameras.xml lists cameras in the same order the
    # frames were written, and only aligned ones (those with a <transform>) appear.
    labels = None
    xml = scene_dir / "cameras.xml"
    if xml.is_file():
        root = ET.parse(xml).getroot()
        labels = [c.get("label") for c in root.iter("camera") if c.find("transform") is not None]
        if len(labels) != len(frames):
            return f"FAIL: {len(frames)} frames but {len(labels)} aligned cameras in cameras.xml"

    new_paths = []
    for i, frame in enumerate(frames):
        m = re.search(r"(\d+)(?=\.[^.]+$)", frame["file_path"])
        if m is None:
            return f"FAIL: cannot parse an index out of {frame['file_path']}"
        idx = int(m.group(1))
        if not 1 <= idx <= len(originals):
            return f"FAIL: index {idx} outside 1..{len(originals)} images"
        original = originals[idx - 1]
        if labels is not None and Path(original).stem != labels[i]:
            return f"FAIL: frame {i} maps to {original} but cameras.xml says {labels[i]}"
        new_paths.append(f"images/{original}")

    if len(set(new_paths)) != len(new_paths):
        return "FAIL: inversion produced duplicate filenames"

    if dry_run:
        return f"would fix {len(frames)} frames (e.g. {frames[0]['file_path']} -> {new_paths[0]})"

    backup = tj.with_suffix(".json.orig")
    if not backup.exists():
        shutil.copy2(tj, backup)
    for frame, p in zip(frames, new_paths):
        frame["file_path"] = p
    tj.write_text(json.dumps(meta, indent=2))

    n_test = sum(1 for p in new_paths if Path(p).name.startswith("test_"))
    return f"fixed {len(frames)} frames ({len(frames) - n_test} frame_ / {n_test} test_)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene_dirs", nargs="+")
    parser.add_argument("--dry-run", "-n", action="store_true")
    args = parser.parse_args()

    failures = 0
    for d in args.scene_dirs:
        result = restore(Path(d.rstrip("/")), args.dry_run)
        print(f"  {d.rstrip('/'):45s} {result}")
        failures += result.startswith("FAIL")
    print(f"----\n{len(args.scene_dirs)} scene(s), {failures} failure(s)")


if __name__ == "__main__":
    main()
