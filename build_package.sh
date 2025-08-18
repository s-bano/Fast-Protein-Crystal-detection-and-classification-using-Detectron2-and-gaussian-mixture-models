#!/bin/bash

# ==============================
# Liste des fichiers à inclure (juste les basenames)
# ==============================
FILES=("gmm_tools.py")

# Dossier de sortie
BUILD_DIR="dist_package"
PACKAGE_DIR="crystalmetrics"

# Nettoyage ancien build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/$PACKAGE_DIR"

# ==============================
# Copier uniquement les fichiers spécifiés
# ==============================
for fname in "${FILES[@]}"; do
    file=$(find . -type f -name "$fname" | head -n 1)
    if [[ -n "$file" ]]; then
        cp "$file" "$BUILD_DIR/$PACKAGE_DIR/"
        echo "✅ Copié $file -> $BUILD_DIR/$PACKAGE_DIR/"
    else
        echo "⚠️ Fichier $fname non trouvé"
    fi
done

# ==============================
# Gérer le README
# ==============================
README_SRC=$(find . -type f -iname "README_PIP.md" | head -n 1)
if [[ -n "$README_SRC" ]]; then
    cp "$README_SRC" "$BUILD_DIR/README.md"
    echo "✅ Copié et renommé $README_SRC -> $BUILD_DIR/README.md"
fi

# ==============================
# Copier pyproject.toml s'il existe
# ==============================
if [[ -f "pyproject.toml" ]]; then
    cp "pyproject.toml" "$BUILD_DIR/"
    echo "✅ Copié pyproject.toml -> $BUILD_DIR/"
fi

echo "🎉 Build prêt dans $BUILD_DIR"
echo "➡️ Pour builder : cd $BUILD_DIR && python -m build"