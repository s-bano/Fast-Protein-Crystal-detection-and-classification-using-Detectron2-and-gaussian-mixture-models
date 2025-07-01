#!/bin/bash

# Usage: ./build_detectron_dataset.sh path/to/json_folder path/to/images_folder path/to/output_dataset_folder

set -e

JSON_DIR="$1"
IMAGES_DIR="$2"
OUTPUT_DIR="$3"

mkdir -p "$OUTPUT_DIR/train"
mkdir -p "$OUTPUT_DIR/val"

echo "Copie des images..."

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

echo "Dataset construit dans $OUTPUT_DIR avec $train_count images dans train et $((total - train_count)) dans val."

# Création des fichiers annotations.json au format coco

echo "Creation du fichier coco pour la base d'entrainement..."
python labelme2coco.py $1 $OUTPUT_DIR/train $OUTPUT_DIR/train/annotations.json --category crystal
echo "Fichier coco d'entrainement cree avec succes!"

echo "Creation du fichier coco pour la base de validation..."
python labelme2coco.py $1 $OUTPUT_DIR/val $OUTPUT_DIR/val/annotations.json --category crystal
echo "Fichier coco de validation cree avec succes!"

# Compression au format zip
echo "Compression du dataset en cours..."
zip -r "${OUTPUT_DIR}.zip" "$OUTPUT_DIR"
echo "Compression terminée : ${OUTPUT_DIR}.zip"
