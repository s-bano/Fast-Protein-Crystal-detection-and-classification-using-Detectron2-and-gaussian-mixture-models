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
- Saves the preprocessed tensor and the transformation parameters in a specified output directory.

Usage:
    python preprocess_image.py /path/to/input_image.jpg [output_directory]

Arguments:
    The script expects:
    - A positional argument for the path to the input image file.
    - An optional second positional argument for the output directory (default: current directory).

Output:
    Saves:
    - A .pt file containing the preprocessed tensor.
    - A .json file containing the transformation parameters.

Note:
- The transformation parameters (scale factor, original size, padding) are needed during post-processing
  to map model predictions back to the original image coordinates.
- Designed for single-class Mask R-CNN models with input size 1024x1024 by default.

Author: Raphaël Kuhn
"""

import cv2
import numpy as np
import torch
import sys
import os
import json

def resize_and_pad_with_info(img, target_size=1024):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    transform_info = {
        "scale": scale,
        "original_size": [h, w],
        "resized_size": [new_h, new_w],
        "padding": [top, bottom, left, right]
    }

    return padded, transform_info

def preprocess_image(img_path, target_size=1024):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Impossible de lire l'image {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, transform_info = resize_and_pad_with_info(img, target_size=target_size)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(img).unsqueeze(0)

    return tensor, transform_info

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py /path/to/image.jpg [output_directory]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    tensor_image, info = preprocess_image(image_path)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    tensor_path = os.path.join(output_dir, base_name + ".pt")
    json_path = os.path.join(output_dir, base_name + ".json")

    torch.save(tensor_image, tensor_path)
    with open(json_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"Tenseur image sauvegardé : {tensor_path}")
    print(f"Transformation info sauvegardée : {json_path}")