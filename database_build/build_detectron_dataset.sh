#!/bin/bash

# Usage:
# ./build_detectron_dataset.sh path/to/json_folder path/to/images_folder path/to/output_dataset_folder

set -e

JSON_DIR="$1"
IMAGES_DIR="$2"
OUTPUT_DIR="$3"

if [ -z "$JSON_DIR" ] || [ -z "$IMAGES_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 path/to/json_folder path/to/images_folder path/to/output_dataset_folder"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/train"
mkdir -p "$OUTPUT_DIR/val"

echo "📂 Copie des images correspondantes..."

# Crée une liste des images ayant un JSON correspondant
matching_images=()
for json_file in "$JSON_DIR"/*.json; do
    base_name=$(basename "$json_file" .json)
    # Cherche une image correspondante avec extensions courantes
    for ext in jpg png jpeg; do
        img_file="$IMAGES_DIR/$base_name.$ext"
        if [ -f "$img_file" ]; then
            matching_images+=("$img_file")
            break
        fi
    done
done

total=${#matching_images[@]}
if [ "$total" -eq 0 ]; then
    echo "❌ Aucune image correspondante trouvée dans $IMAGES_DIR pour les JSON de $JSON_DIR."
    exit 1
fi

train_count=$((total * 70 / 100))

# Mélange aléatoirement
SHUF_COMMAND=$(command -v gshuf || command -v shuf)
shuf_list=($($SHUF_COMMAND -e "${matching_images[@]}"))

# Copie les images dans train et val
for i in "${!shuf_list[@]}"; do
    img_path="${shuf_list[$i]}"
    if [ "$i" -lt "$train_count" ]; then
        cp "$img_path" "$OUTPUT_DIR/train/"
    else
        cp "$img_path" "$OUTPUT_DIR/val/"
    fi
done

echo "✅ Dataset construit dans $OUTPUT_DIR avec:"
echo "   - $train_count images dans train"
echo "   - $((total - train_count)) images dans val"

# Création des fichiers annotations.json au format coco
echo "📝 Création du fichier COCO pour l'ensemble d'entraînement..."
python labelme2coco.py "$JSON_DIR" "$OUTPUT_DIR/train" "$OUTPUT_DIR/train/annotations.json"
echo "✅ Fichier COCO d'entraînement créé avec succès !"

echo "📝 Création du fichier COCO pour l'ensemble de validation..."
python labelme2coco.py "$JSON_DIR" "$OUTPUT_DIR/val" "$OUTPUT_DIR/val/annotations.json"
echo "✅ Fichier COCO de validation créé avec succès !"

# Compression en zip
echo "🗜️  Compression du dataset en cours..."
zip -r "${OUTPUT_DIR}.zip" "$OUTPUT_DIR"
echo "✅ Compression terminée : ${OUTPUT_DIR}.zip"