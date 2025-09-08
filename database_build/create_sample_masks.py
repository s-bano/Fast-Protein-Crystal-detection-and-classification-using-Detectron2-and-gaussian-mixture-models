"""
Genere les masques de segmentation individuels et combines a partir d une image et de son json labelme

Usage: python create_sample_masks.py path/image path/json

"""




import os
import json
import cv2, sys
import numpy as np
from PIL import Image


OUTPUT_DIR = "masks"                       # dossier où les masques seront sauvegardés

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fonction pour créer des masques à partir d'une image et son JSON
def create_masks(image_path, annotation_path):
    # Charger l'image pour récupérer sa taille
    img = Image.open(image_path)
    width, height = img.size

    # Masque combiné
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Charger les annotations
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    for idx, shape in enumerate(data.get("shapes", [])):
        points = np.array(shape["points"], dtype=np.int32)
        individual_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Remplir le polygone correspondant à l'objet
        cv2.fillPoly(individual_mask, [points], 255)
        
        # Sauvegarder le masque individuel
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask_{idx}.png")
        cv2.imwrite(mask_path, individual_mask)
        
        # Ajouter au masque combiné
        combined_mask = cv2.bitwise_or(combined_mask, individual_mask)

    # Sauvegarder le masque combiné
    combined_mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask_combined.png")
    cv2.imwrite(combined_mask_path, combined_mask)
    print(f"Masques créés pour {image_path}")




if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Erroro: mauvais usage")
    else:
        create_masks(sys.argv[1], sys.argv[2])