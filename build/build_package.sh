#!/bin/bash

# === Liste des fichiers "model" à copier ===
FILES_MODEL=(
    "./Classification/scaler_classif_0908.joblib"
    "./Classification/model_classif_0908.joblib"
    "./AI/models/model_0804.pth"
)

# === Noms cibles correspondants ===
FILES_MODEL_RENAME=(
    "scaler_classif.joblib"
    "model_classif.joblib"
    "model_detectron.pth"
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

rm "./package&model.zip"

# Créer les sous-dossiers
mkdir -p "$DEST/models"
mkdir -p "$DEST/aicm"

# Copier les fichiers modèles avec renommage
for i in "${!FILES_MODEL[@]}"; do
    SRC="${FILES_MODEL[$i]}"
    DEST_NAME="${FILES_MODEL_RENAME[$i]}"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$DEST/models/$DEST_NAME"
        echo "✅ Copié (model) : $SRC → $DEST_NAME"
    else
        echo "⚠️ Fichier modèle introuvable : $SRC"
    fi
done

# Copier les fichiers scripts (même nom)
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