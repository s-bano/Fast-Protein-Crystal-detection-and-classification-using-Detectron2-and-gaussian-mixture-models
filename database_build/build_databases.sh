#!/bin/bash

# Usage: ./build_datasets.sh -u   → pour U-Net
#        ./build_datasets.sh -r   → pour Mask R-CNN

# Dossiers à adapter si besoin
JSON_DIR="jsons"
IMAGE_DIR="images"
MULTI_MASKS_DIR="multi_masks"
SINGLE_MASKS_DIR="single_masks"

usage() {
    echo "Usage: $0 [-u] [-r]"
    echo "  -u    Générer le dataset pour U-Net"
    echo "  -r    Générer le dataset pour Mask R-CNN"
    exit 1
}

# Si aucun argument n’est donné
if [ $# -eq 0 ]; then
    usage
fi

while getopts ":ur" opt; do
  case $opt in
    u)
      echo "🔧 Génération du dataset pour U-Net..."
      python json2masks.py "$JSON_DIR" "$IMAGE_DIR" "$MULTI_MASKS_DIR" || exit 1
      python masks2single.py "$MULTI_MASKS_DIR" "$SINGLE_MASKS_DIR" || exit 1
      python build_u-net_dataset.py "$IMAGE_DIR" "$SINGLE_MASKS_DIR" || exit 1
      echo "✅ Dataset U-Net généré."
      latest_zip=$(ls -t *_dataset_*.zip 2>/dev/null | head -n 1)
      if [[ -n "$latest_zip" ]]; then
        count="${latest_zip##*_}"
        count="${count%.zip}"
        echo "📦 $count images dans le dataset : $latest_zip"
      fi
      ;;
    r)
      echo "🔧 Génération du dataset pour Mask R-CNN..."
      python json2masks.py "$JSON_DIR" "$IMAGE_DIR" "$MULTI_MASKS_DIR" || exit 1
      python build_r-cnn_dataset.py "$IMAGE_DIR" "$MULTI_MASKS_DIR" || exit 1
      echo "✅ Dataset Mask R-CNN généré."
      latest_zip=$(ls -t *_dataset_*.zip 2>/dev/null | head -n 1)
      if [[ -n "$latest_zip" ]]; then
        count="${latest_zip##*_}"
        count="${count%.zip}"
        echo "📦 $count images dans le dataset : $latest_zip"
      fi
      ;;
    \?)
      echo "❌ Option invalide : -$OPTARG" >&2
      usage
      ;;
  esac
done