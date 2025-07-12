"""
Postprocessing Script for Mask R-CNN TorchScript Model Output

This script applies the inverse transformations recorded during preprocessing
(back-scaling and unpadding) to the predicted mask produced by the Mask R-CNN model.

Key Features:
- Loads a predicted mask tensor saved in .pt format.
- Loads the corresponding transformation info in .json format.
- Applies inverse padding removal and resizing to map the mask back to the original image size.
- Saves the resulting mask as a .png image with the same base name as the input mask.

Usage:
    python postprocess_mask.py predicted_mask.pt transform_info.json [output_directory]

Arguments:
    - predicted_mask.pt: Path to the predicted mask tensor.
    - transform_info.json: Path to the JSON file containing preprocessing transformation info.
    - output_directory (optional): Directory where the output mask will be saved (default: current directory).

Output:
    - A .png file of the mask mapped back to the original image size.

Author: Raphaël Kuhn
"""

import torch
import json
import sys
import os
import numpy as np
import cv2

def inverse_transform_mask(mask_tensor, transform_info):
    """
    Apply inverse padding removal and resizing to the predicted mask tensor.
    """
    mask = mask_tensor.squeeze().cpu().numpy()

    # Convert to uint8 binary mask if needed
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255

    # Remove padding
    top, bottom, left, right = transform_info["padding"]
    h_padded, w_padded = mask.shape[:2]
    mask_cropped = mask[top:h_padded - bottom, left:w_padded - right]

    # Resize back to original size
    original_h, original_w = transform_info["original_size"]
    mask_resized = cv2.resize(mask_cropped, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    return mask_resized

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python postprocess_mask.py predicted_mask.pt transform_info.json [output_directory]")
        sys.exit(1)

    mask_path = sys.argv[1]
    transform_json_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Load mask tensor
    mask_tensor = torch.load(mask_path, map_location="cpu")

    # Load transformation info
    with open(transform_json_path, "r") as f:
        transform_info = json.load(f)

    # Apply inverse transformations
    mask_image = inverse_transform_mask(mask_tensor, transform_info)

    # Prepare output path
    base_name = os.path.splitext(os.path.basename(mask_path))[0]
    output_path = os.path.join(output_dir, base_name + "_postprocessed.png")

    # Save mask
    cv2.imwrite(output_path, mask_image)

    print(f"Masque post-traité sauvegardé : {output_path}")
    print(f"Shape du masque post-traité : {mask_image.shape}")