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

def labelme_json_to_instance_masks(json_path):
    json_path = Path(json_path)
    with open(json_path) as f:
        data = json.load(f)

    # Lire les dimensions de l'image associée
    img_path = json_path.with_suffix('.jpg')
    if not img_path.exists():
        img_path = json_path.with_suffix('.png')
    assert img_path.exists(), f"Image {img_path} non trouvée"

    image = Image.open(img_path)
    img_shape = np.array(image).shape  # (H, W, C)

    shapes = data["shapes"]
    output_dir = json_path.stem + "_masks"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Création des masques dans : {output_dir}")
    for i, shape in enumerate(shapes):
        points = shape["points"]
        mask = polygon_to_mask(img_shape, points)
        mask_filename = f"mask_{i+1:03d}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        Image.fromarray(mask).save(mask_path)
        print(f"✔️ {mask_filename}")

# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Utilisation : python labelme_to_instance_masks.py fichier.json")
    else:
        labelme_json_to_instance_masks(sys.argv[1])