'''
Ce script deplace tout les fichiers .json d un dossier dans un autre
Utile dans le contexte de labelisation des images
'''

import os
import shutil

def main(dossier_source, dossier_destination):

    # S'assurer que le dossier de destination existe
    os.makedirs(dossier_destination, exist_ok=True)

    # Parcours des fichiers dans le dossier source
    for nom_fichier in os.listdir(dossier_source):
        if nom_fichier.endswith(".json"):
            chemin_source = os.path.join(dossier_source, nom_fichier)
            chemin_destination = os.path.join(dossier_destination, nom_fichier)
            shutil.move(chemin_source, chemin_destination)
            print(f"Déplacé : {nom_fichier}")
        
# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Utilisation : python move_json.py dossier_source dossier_destination")
    else:
        main(sys.argv[1], sys.argv[2])