import json
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

def polygon_to_mask(img_shape, points):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    points_array = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, points_array, 255)
    return mask

def labelme_json_to_instance_masks(json_path, images_dir, output_root):
    json_path = Path(json_path)
    image_stem = json_path.stem
    images_dir = Path(images_dir)

    # Cherche l'image correspondante dans le dossier des images
    possible_extensions = ['.png', '.jpg', '.jpeg']
    img_path = None
    for ext in possible_extensions:
        candidate = images_dir / f"{image_stem}{ext}"
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        raise FileNotFoundError(f"Image for {json_path.name} not found in {images_dir}")

    with open(json_path) as f:
        data = json.load(f)

    image = Image.open(img_path)
    img_shape = np.array(image).shape  # (H, W, C)

    shapes = data["shapes"]
    output_dir = Path(output_root) / image_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating masks for: {json_path.name}")
    for i, shape in enumerate(shapes):
        points = shape["points"]
        mask = polygon_to_mask(img_shape, points)
        mask_filename = f"mask_{i+1:03d}.png"
        mask_path = output_dir / mask_filename
        Image.fromarray(mask).save(mask_path)
        print(f"✔️ {mask_filename}")

def process_all_jsons(json_dir, images_dir, output_dir):
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print("❌ No JSON files found in input directory.")
        return

    for json_file in json_files:
        try:
            labelme_json_to_instance_masks(json_file, images_dir, output_dir)
        except Exception as e:
            print(f"⚠️ Failed to process {json_file.name}: {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python json2masks.py json_folder images_folder output_folder")
    else:
        process_all_jsons(sys.argv[1], sys.argv[2], sys.argv[3])