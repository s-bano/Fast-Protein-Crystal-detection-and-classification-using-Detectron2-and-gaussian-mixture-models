import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from labelme.cli import export_json  # import de la fonction interne

def json_to_binary_mask(json_path, output_dir):
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dossier temporaire pour les fichiers générés
    tmp_dir = json_path.stem + "_json"
    tmp_path = Path(tmp_dir)

    try:
        # Exporte le dataset localement (label.png, etc.)
        export_json.main([str(json_path)])

        # Charge le masque
        label = np.array(Image.open(tmp_path / "label.png"))
        binary_mask = (label > 0).astype(np.uint8) * 255

        # Sauvegarde du masque binaire
        out_name = json_path.stem + "_mask.png"
        Image.fromarray(binary_mask).save(output_dir / out_name)
        print(f"✔ {out_name}")

    finally:
        # Nettoie le dossier temporaire
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

def process_all_jsons(json_dir, output_dir):
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)

    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print("❌ No JSON files found.")
        return

    for json_file in json_files:
        try:
            json_to_binary_mask(json_file, output_dir)
        except Exception as e:
            print(f"⚠️ Failed to process {json_file.name}: {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python json2singleMask.py json_folder output_folder")
    else:
        process_all_jsons(sys.argv[1], sys.argv[2])