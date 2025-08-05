#!/bin/bash

# Crée les dossiers si besoin
mkdir -p train val

# Liste toutes les images (jpg, png, jpeg, etc.)
images=(*.jpg *.jpeg *.png *.bmp *.gif)

# Filtre les fichiers inexistants (si aucun ne matche)
images=( "${images[@]}" )
if [ ${#images[@]} -eq 0 ]; then
  echo "Aucune image trouvée dans le dossier courant."
  exit 1
fi

# Mélange les fichiers aléatoirement
shuffled=($(printf "%s\n" "${images[@]}" | shuf))

# Calcul du nombre d'images à mettre dans train (80%)
total=${#shuffled[@]}
train_count=$(( total * 80 / 100 ))

# Répartition
for i in "${!shuffled[@]}"; do
  file="${shuffled[$i]}"
  if [ $i -lt $train_count ]; then
    cp "$file" train/
  else
    cp "$file" val/
  fi
done

echo "Répartition terminée :"
echo "  ${train_count} images -> train/"
echo "  $((total - train_count)) images -> val/"
