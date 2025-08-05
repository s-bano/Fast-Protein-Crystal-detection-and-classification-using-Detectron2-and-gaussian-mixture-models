"""
Usage : python3 annotator.py <dossier_images>"
"""


import os
import json
import sys
from PIL import Image

# Vérification des arguments
if len(sys.argv) != 2:
    print("Utilisation : python3 annotator.py <dossier_images>")
    sys.exit(1)

image_dir = sys.argv[1]
if not os.path.isdir(image_dir):
    print(f"Erreur : '{image_dir}' n'est pas un dossier valide.")
    sys.exit(1)

# Nom du fichier de sortie
output_json = os.path.join(image_dir, "annotations.json")

# Extensions d'images autorisées
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# Liste des images
images = []
image_id = 1

for fname in sorted(os.listdir(image_dir)):
    if fname.lower().endswith(valid_exts):
        image_path = os.path.join(image_dir, fname)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Erreur lors de l'ouverture de {fname}: {e}")
            continue
        images.append({
            "id": image_id,
            "file_name": fname,
            "width": width,
            "height": height
        })
        image_id += 1

# Format COCO minimal
coco_data = {
    "info": {
        "description": "Crsital Dataset",
        "version": "1.0",
        "year": 2025
    },
    "licenses": [],
    "images": images,
    "annotations": [],
    "categories": [
    {
      "id": 1,
      "name": "cristal"
    },
    {
      "id": 2,
      "name": "non-cristal"
    }
  ]
}

# Écriture du fichier JSON
with open(output_json, "w") as f:
    json.dump(coco_data, f, indent=2)

print(f"Fichier '{output_json}' généré avec {len(images)} images.")
