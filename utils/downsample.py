import argparse
from pathlib import Path
from PIL import Image

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def downsample_folder(input_dir: Path, scale: int, output_dir: Path) -> int:
    """Downsample every image in input_dir by integer scale and save to output_dir.

    Args:
        input_dir: folder with original images
        scale: integer downsample factor (e.g., 2, 4, 8)
        output_dir: destination folder for downsampled images

    Returns:
        Number of processed images.
    """
    if scale < 1:
        raise ValueError("Scale must be >= 1")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return 0

    processed = 0
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Warning: could not open {img_path.name}: {e}")
            continue

        w, h = img.size
        new_w = max(1, w // scale)
        new_h = max(1, h // scale)

        img_small = img.resize((new_w, new_h), resample=Image.LANCZOS)

        out_path = output_dir / img_path.name
        img_small.save(out_path)
        print(f"Saved downsampled image: {out_path}")
        processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser(description="Downsample all images in a folder by an integer scale.")
    parser.add_argument("input_dir", help="Folder containing input images")
    parser.add_argument("scale", type=int, help="Downsample factor (integer >=1), e.g., 2,4,8")
    parser.add_argument("--output_path", default=None, help="Folder to save downsampled images (defaults to ./downsampled)")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_path) if args.output_path else input_dir.parent / (input_dir.name + "_downsampled")

    num = downsample_folder(input_dir, args.scale, output_dir)
    print(f"Processed {num} image(s).")


if __name__ == "__main__":
    main()
