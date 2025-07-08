"""
Image Preprocessing Script for Detectron2 TorchScript Model Inference

This script performs the necessary preprocessing steps on input images to prepare them
for inference with a Detectron2 Mask R-CNN model exported to TorchScript.

Key Features:
- Loads an image from a given file path.
- Converts the image from BGR (OpenCV default) to RGB format.
- Resizes the image while maintaining the aspect ratio, scaling the largest dimension to a target size.
- Pads the resized image to obtain a square image of the specified target size, with zero-padding (black).
- Normalizes pixel values to the [0, 1] range.
- Converts the image to a PyTorch tensor with shape [1, 3, target_size, target_size] suitable for model input.
- Returns both the preprocessed tensor and the transformation parameters needed for post-processing.

Usage:
    python preprocess_image.py /path/to/input_image.jpg

Arguments:
    The script expects a single positional argument which is the path to the input image file.

Output:
    Prints the shape of the preprocessed tensor and the details of the transformations applied.

Note:
- The transformation parameters (scale factor, original size, padding) should be saved and
  used later during post-processing to map model predictions back to the original image coordinates.
- This preprocessing matches the fixed input size requirement of the exported TorchScript model.
- Designed for single-class Mask R-CNN models with input size 1024x1024 by default,
  but target size can be adjusted in the script.

Author: Raphaël Kuhn
"""

import cv2
import numpy as np
import torch
import sys

def resize_and_pad_with_info(img, target_size=1024):
    """
    Redimensionne l'image en conservant le ratio et ajoute du padding pour obtenir un carré target_size x target_size.
    Retourne l'image transformée et un dictionnaire d'infos pour inversion en post-traitement.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]  # padding noir
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    transform_info = {
        "scale": scale,
        "original_size": (h, w),
        "resized_size": (new_h, new_w),
        "padding": (top, bottom, left, right)
    }

    return padded, transform_info

def preprocess_image(img_path, target_size=1024):
    """
    Charge l'image, la convertit en RGB, applique resize+padding, normalise et convertit en tensor PyTorch.
    Retourne le tenseur image et les infos de transformation.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, transform_info = resize_and_pad_with_info(img, target_size=target_size)
    img = img.astype(np.float32) / 255.0  # normalisation [0,1]

    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(img).unsqueeze(0)  # batch dimension

    return tensor, transform_info



# Exemple d'utilisation modifié
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ton_script.py chemin/vers/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    tensor_image, info = preprocess_image(image_path)
    print(f"Tenseur image shape: {tensor_image.shape}")  # torch.Size([1, 3, 1024, 1024])
    print(f"Transformation info: {info}")