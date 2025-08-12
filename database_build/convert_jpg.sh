#!/bin/bash

# Dossier où se trouvent les images (par défaut : dossier courant)
INPUT_DIR="output_more_images"
OUTPUT_DIR="converted_jpg"

# Créer le dossier de sortie s'il n'existe pas
mkdir -p "$OUTPUT_DIR"

# Vérifier si ImageMagick est installé
if ! command -v convert &> /dev/null; then
    echo "ImageMagick n'est pas installé. Installe-le avec : brew install imagemagick"
    exit 1
fi

# Boucle sur tous les fichiers .tif et .tiff
for img in "$INPUT_DIR"/*.tif "$INPUT_DIR"/*.tiff; do
    # Vérifier que le fichier existe pour éviter les erreurs
    [ -e "$img" ] || continue

    # Nom de fichier sans extension
    filename=$(basename "$img")
    filename_noext="${filename%.*}"

    # Conversion en JPG
    magick "$img" "$OUTPUT_DIR/$filename_noext.jpg"
    echo "Converti : $img → $OUTPUT_DIR/$filename_noext.jpg"
done

echo "✅ Conversion terminée. Fichiers JPG dans : $OUTPUT_DIR"