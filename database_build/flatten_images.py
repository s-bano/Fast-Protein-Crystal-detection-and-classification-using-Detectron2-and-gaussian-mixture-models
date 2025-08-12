'''
This script reorganizes all images from a folder and its subfolders into a single output directory.  
It also renames the images to ensure a consistent and uniform naming convention.  
Note: Files are copied**, not moved, so the original folder structure remains unchanged.

Usage : python flatten_names.py [source_folder] [output_folder]
'''

import os
import shutil
from pathlib import Path

valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}

def flatten_and_rename_images(input_dir, output_dir, extensions=valid_exts):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 1013
    for filepath in sorted(input_dir.rglob('*')):
        if filepath.suffix.lower() in extensions:
            count += 1
            new_filename = f"image_{count:04d}{filepath.suffix.lower()}"
            destination = output_dir / new_filename
            shutil.copy2(filepath, destination)
            print(f"✔ Copié : {filepath} → {destination}")

    print(f"\n✅ Total images copiées et renommées : {count}")

# Exemple d'utilisation :
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage : python flatten_names.py [source_folder] [output_folder]")
    else:
        flatten_and_rename_images(sys.argv[1], sys.argv[2])