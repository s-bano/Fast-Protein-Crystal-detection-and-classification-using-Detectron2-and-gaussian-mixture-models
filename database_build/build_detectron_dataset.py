'''
Usage: python build_detectron_dataset.py path/to/json_folder path/to/images_folder path/to/output_dataset_folder
'''

import os
import sys
import shutil
import random
import zipfile
import subprocess
from pathlib import Path

def copy_matching_images(json_dir, images_dir, output_dir):
    output_train = output_dir / "train"
    output_val = output_dir / "val"
    output_train.mkdir(parents=True, exist_ok=True)
    output_val.mkdir(parents=True, exist_ok=True)

    matching_images = []
    for json_file in json_dir.glob("*.json"):
        base_name = json_file.stem
        for ext in ["jpg", "png", "jpeg"]:
            img_file = images_dir / f"{base_name}.{ext}"
            if img_file.exists():
                matching_images.append(img_file)
                break

    if not matching_images:
        print(f"❌ Aucune image correspondante trouvée dans {images_dir} pour les JSON de {json_dir}.")
        sys.exit(1)

    random.shuffle(matching_images)
    train_count = int(len(matching_images) * 0.7)

    for i, img_path in enumerate(matching_images):
        if i < train_count:
            shutil.copy(img_path, output_train)
        else:
            shutil.copy(img_path, output_val)

    print(f"✅ Dataset construit dans {output_dir} avec:")
    print(f"   - {train_count} images dans train")
    print(f"   - {len(matching_images) - train_count} images dans val")

def create_coco_annotations(json_dir, subset_dir):
    annotation_file = subset_dir / "annotations.json"
    print(f"📝 Création du fichier COCO pour {subset_dir.name}...")
    subprocess.run([sys.executable, "labelme2coco.py", str(json_dir), str(subset_dir), str(annotation_file)], check=True)
    print(f"✅ Fichier COCO pour {subset_dir.name} créé avec succès !")

def compress_dataset(output_dir):
    zip_path = output_dir.with_suffix('.zip')
    print("🗜️  Compression du dataset en cours...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = Path(root) / file
                zipf.write(file_path, arcname=file_path.relative_to(output_dir.parent))
    print(f"✅ Compression terminée : {zip_path}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python build_detectron_dataset.py path/to/json_folder path/to/images_folder path/to/output_dataset_folder")
        sys.exit(1)

    json_dir = Path(sys.argv[1])
    images_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    copy_matching_images(json_dir, images_dir, output_dir)
    create_coco_annotations(json_dir, output_dir / "train")
    create_coco_annotations(json_dir, output_dir / "val")
    compress_dataset(output_dir)

if __name__ == "__main__":
    main()