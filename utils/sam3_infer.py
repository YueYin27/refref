import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# Initialize model and processor once
model = build_sam3_image_model()
processor = Sam3Processor(model)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _to_binary_mask(mask_tensor, image_width: int, image_height: int) -> np.ndarray:
    mask = np.squeeze(mask_tensor.cpu().numpy())
    mask = (mask > 0).astype(np.uint8) * 255

    if mask.ndim == 1 and mask.size == image_height * image_width:
        mask = mask.reshape(image_height, image_width)
    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape {mask.shape}, expected 2D.")

    return mask


def sam3_infer_image(image_path: Path, prompt: str = "tabletop") -> np.ndarray:
    """Run SAM3 on one image and return a merged binary mask (0 or 255)."""
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output["masks"]
    image_width, image_height = image.size

    merged_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for mask_tensor in masks:
        mask = _to_binary_mask(mask_tensor, image_width, image_height)
        merged_mask = np.maximum(merged_mask, mask)

    return merged_mask


def sam3_infer_folder(input_dir: Path, prompt: str, output_dir: Path) -> int:
    """Run SAM3 for every image in input_dir and save masks in output_dir."""
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )

    if not image_paths:
        print(f"No images found in {input_dir}")
        return 0

    for image_path in image_paths:
        merged_mask = sam3_infer_image(image_path, prompt=prompt)
        output_path = output_dir / image_path.name
        Image.fromarray(merged_mask).save(output_path)
        print(f"Saved mask: {output_path}")

    return len(image_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM3 inference on all images in a folder.")
    parser.add_argument("image_dir", help="Path to folder containing input images")
    parser.add_argument("--prompt", default="tabletop", help="Text prompt for SAM3")
    parser.add_argument("--output_path", default="masks", help="Output folder for predicted masks")

    args = parser.parse_args()
    num_processed = sam3_infer_folder(
        input_dir=Path(args.image_dir),
        prompt=args.prompt,
        output_dir=Path(args.output_path),
    )
    print(f"Processed {num_processed} image(s).")
