import os
import json
import cv2
import argparse
from glob import glob

def convert_split_to_coco(images_dir, labels_dir, output_json_path, category_name="cristal"):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": category_name}],
    }
    annotation_id = 1
    image_id = 1

    image_paths = sorted(glob(os.path.join(images_dir, "*")))

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(label_path):
            print(f"⚠️ Label not found for {filename}, skipping.")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Cannot read image {img_path}, skipping.")
            continue

        height, width = image.shape[:2]

        coco["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, w, h = map(float, parts)

                x_min = (xc - w / 2) * width
                y_min = (yc - h / 2) * height
                abs_w = w * width
                abs_h = h * height

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x_min, y_min, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0,
                    "segmentation": []
                })
                annotation_id += 1

        image_id += 1

    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"✅ COCO annotations saved to {output_json_path} ({len(coco['annotations'])} annotations)")

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO-format annotations to COCO format.")
    parser.add_argument("--train_images", required=True)
    parser.add_argument("--train_labels", required=True)
    parser.add_argument("--val_images", required=True)
    parser.add_argument("--val_labels", required=True)
    parser.add_argument("--test_images", required=True)
    parser.add_argument("--test_labels", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--category", default="cristal")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    convert_split_to_coco(args.train_images, args.train_labels, os.path.join(args.output_dir, "train.json"), args.category)
    convert_split_to_coco(args.val_images, args.val_labels, os.path.join(args.output_dir, "val.json"), args.category)
    convert_split_to_coco(args.test_images, args.test_labels, os.path.join(args.output_dir, "test.json"), args.category)

if __name__ == "__main__":
    main()