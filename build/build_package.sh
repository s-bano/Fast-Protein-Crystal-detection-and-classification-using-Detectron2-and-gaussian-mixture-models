#!/bin/bash


# === Liste des fichiers "model" à copier ===
FILES_MODEL=(
    "./Classification/gmm/scaler_classif.joblib"
    "./Classification/gmm/model_classif.joblib"
    "./AI/models/model_0804.pth"
)

# === Liste des fichiers "script" à copier ===
FILES_SCRIPT=(
    "./Classification/export_tools.py"
    "./Classification/gmm/gmm_tools.py"
    "./AI/detectron_tools.py"
    "./build/__init__.py"
)

# === Dossier racine de destination ===
DEST="./package&model"

# Créer les sous-dossiers
mkdir -p "$DEST/models"
mkdir -p "$DEST/aicm"

# Copier les fichiers modèles
for FILE in "${FILES_MODEL[@]}"; do
    if [ -f "$FILE" ]; then
        cp "$FILE" "$DEST/models/"
        echo "✅ Copié (model) : $FILE"
    else
        echo "⚠️ Fichier modèle introuvable : $FILE"
    fi
done

# Copier les fichiers scripts
for FILE in "${FILES_SCRIPT[@]}"; do
    if [ -f "$FILE" ]; then
        cp "$FILE" "$DEST/aicm/"
        echo "✅ Copié (script) : $FILE"
    else
        echo "⚠️ Fichier script introuvable : $FILE"
    fi
done

# Créer l’archive zip
zip -r "package&model.zip" "$DEST"
echo "📦 Archive créée : package&model.zip"

# Supprimer le dossier temporaire
rm -rf "$DEST"