import os
import shutil
from pathlib import Path

def flatten_and_rename_images(input_dir, output_dir, extensions={'.jpg', '.jpeg', '.png'}):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
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
        print("Utilisation : python flatten_images_renamed.py dossier_source dossier_destination")
    else:
        flatten_and_rename_images(sys.argv[1], sys.argv[2])