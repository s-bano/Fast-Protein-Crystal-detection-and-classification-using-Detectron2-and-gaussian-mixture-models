#!/usr/bin/env python3

"""
This script converts a folder of Labelme annotations (.json files) into a single COCO-format JSON file
suitable for training object detection or instance segmentation models (e.g., Mask R-CNN).

Each JSON file must contain polygon annotations (not bounding boxes) and reference an image file 
present in a specified image directory. The output file will contain bounding boxes and segmentation masks
for each annotated object, along with the required image and category metadata in COCO format.

Usage:
    python labelme_to_coco.py <json_directory> <image_directory> <output_json_file> [--category <class_name>]

Arguments:
    <json_directory>     Path to the folder containing Labelme .json files.
    <image_directory>    Path to the folder containing corresponding images.
    <output_json_file>   Desired path for the generated COCO-format .json file.
    --category           Optional: name of the object class (default: "cristal").

Example:
    python labelme2coco.py ./jsons ./images output.json --category crystal

Requirements:
    - Python 3
    - OpenCV (cv2)
    - NumPy

Author: Raphaël Kuhn
"""

import json
import os
import cv2
import numpy as np
import argparse
from glob import glob

def convert_labelme_to_coco(json_dir, image_dir, output_path, category_name="cristal"):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": category_name}],
    }

    annotation_id = 1
    json_files = glob(os.path.join(json_dir, "*.json"))

    for image_id, json_file in enumerate(json_files, start=1):
        with open(json_file, 'r') as f:
            data = json.load(f)

        image_filename = data["imagePath"]
        image_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"⚠️ Image manquante : {image_filename}, ignorée")
            continue

        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        coco_output["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height
        })

        for shape in data["shapes"]:
            points = shape["points"]
            segmentation = [np.array(points).flatten().tolist()]
            polygon = np.array(points)
            x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))

            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x, y, w, h],
                "segmentation": segmentation,
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    print(f"✅ Conversion terminée : {output_path} ({len(coco_output['annotations'])} annotations)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Labelme JSONs to COCO format.")
    parser.add_argument("json_dir", help="Dossier contenant les fichiers .json de Labelme")
    parser.add_argument("image_dir", help="Dossier contenant les images correspondantes")
    parser.add_argument("output_json", help="Fichier de sortie COCO .json")
    parser.add_argument("--category", default="cristal", help="Nom de la classe (défaut: 'cristal')")
    args = parser.parse_args()

    convert_labelme_to_coco(args.json_dir, args.image_dir, args.output_json, args.category)