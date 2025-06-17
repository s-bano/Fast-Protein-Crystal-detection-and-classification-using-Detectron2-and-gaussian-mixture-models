#!/bin/bash

# Dossier source contenant les images
SOURCE_DIR="extracted_cristals"
cd "$SOURCE_DIR" || exit 1

# Dossiers de destination
mkdir -p ../train ../val ../test

# Récupérer tous les fichiers images (jpg, jpeg, png)
files=( $(find . -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | shuf) )

total=${#files[@]}
if [ "$total" -eq 0 ]; then
    echo "Aucune image trouvée dans $SOURCE_DIR."
    exit 1
fi

# Proportions
train_end=$(( total * 70 / 100 ))
val_end=$(( total * 85 / 100 ))

for i in "${!files[@]}"; do
    file="${files[$i]}"
    clean_file="${file#./}"  # supprimer le ./ au début
    if [ "$i" -lt "$train_end" ]; then
        mv "$clean_file" ../train/
    elif [ "$i" -lt "$val_end" ]; then
        mv "$clean_file" ../val/
    else
        mv "$clean_file" ../test/
    fi
done

# Résumé
echo "Répartition terminée :"
echo "- train : $train_end fichiers"
echo "- val   : $((val_end - train_end)) fichiers"
echo "- test  : $((total - val_end)) fichiers"