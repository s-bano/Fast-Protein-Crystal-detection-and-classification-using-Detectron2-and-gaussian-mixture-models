"""
Le but de ce script est de combiner tous les fichiers CSV contenus dans un dossier récursif
en un seul fichier Excel, avec chaque CSV dans une feuille distincte et son chemin relatif
au-dessus des titres de colonnes.
"""

import os
import pandas as pd
from openpyxl import Workbook

# Spécifie le dossier de base à parcourir
base_dir = "/Users/pagatok/Desktop/images_output_test5"

# Fichier de sortie
output_excel = "combined_output.xlsx"

# Crée un writer pour Excel
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)

                    # Calcule le chemin relatif à base_dir
                    relative_path = os.path.relpath(file_path, base_dir)

                    # Crée une ligne unique avec le chemin relatif au-dessus des titres
                    empty_row = [''] * df.shape[1]
                    path_row = [f"{relative_path}"] + empty_row[1:]

                    # Insère le chemin en première ligne, puis les colonnes et les données
                    df_with_path = pd.concat([
                        pd.DataFrame([path_row], columns=df.columns),
                        df
                    ], ignore_index=True)

                    # Nom de feuille Excel (max 31 caractères)
                    sheet_name = os.path.splitext(file)[0][:31]
                    sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '_')).rstrip()

                    df_with_path.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"Ajouté: {relative_path} -> Onglet: {sheet_name}")
                except Exception as e:
                    print(f"Erreur lors de la lecture de {file_path}: {e}")

print(f"\n✅ Tous les fichiers CSV ont été combinés dans {output_excel}")