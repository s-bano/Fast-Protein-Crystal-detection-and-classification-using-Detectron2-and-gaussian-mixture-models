import json, os, sys, cv2, zipfile
from PIL import Image


# Cree un fichier temporaire annotations.json 
def json_annotator(image_dir):
    
    if os.path.exists(os.path.join(image_dir, "annotations.json")): 
        print("Found an existing annotations.json")
        return
    
    print("Creating annotations.json...")
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')  # Extensions d'images autorisées
    image_id = 1 

    if not os.path.isdir(image_dir):
        print(f"Erreur : '{image_dir}' n'est pas un dossier valide.")
        sys.exit(1)
            
    # Nom du fichier de sortie
    output_json = os.path.join(image_dir, "annotations.json")

    # Liste des images
    images = []

    # Parcours récursif avec os.walk
    for root, dirs, files in os.walk(image_dir):
        for fname in sorted(files):
            if fname.lower().endswith(valid_exts) and "_MACOSX" not in fname:
                image_path = os.path.join(root, fname)
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"Erreur lors de l'ouverture {image_path}: {e}")
                    continue
                # On stocke le chemin relatif pour plus de portabilité
                rel_path = os.path.relpath(image_path, image_dir)
                images.append({
                    "id": image_id,
                    "file_name": rel_path,
                    "width": width,
                    "height": height
                })
                image_id += 1

    # Format COCO minimal
    coco_data = {
        "info": {
            "description": "Cristal Dataset",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [],
        "images": images,
        "annotations": [],
        "categories": [
            {"id": 1, "name": "cristal"},
            {"id": 2, "name": "non-cristal"}
        ]
    }

    # Écriture du fichier JSON
    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"Fichier '{output_json}' généré avec {len(images)} images.")
    
    return


# Pre-process resizing 
def resize_to_multiple_of_32(img):
    h, w = img.shape[:2]
    new_h = (h // 32) * 32
    new_w = (w // 32) * 32
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


# Cette focntion se charge 
def zip_handle(zip_path, extract_dir="/content/extracted"):
    """
    Extrait un zip et retourne le chemin du dossier contenant les fichiers principaux.
    
    Si le zip contient un seul dossier à l'intérieur, retourne ce dossier.
    Sinon, retourne le dossier d'extraction.
    """
    os.makedirs(extract_dir, exist_ok=True)

    # Extraction
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Lister le contenu
    # items = os.listdir(extract_dir)
    # if len(items) == 1:
    #     single_item_path = os.path.join(extract_dir, items[0])
    #     if os.path.isdir(single_item_path):
    #         return single_item_path

    # Sinon retourner le dossier d'extraction
    return extract_dir