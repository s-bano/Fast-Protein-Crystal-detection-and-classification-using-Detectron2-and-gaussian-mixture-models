import sys
import os
import zipfile


def compter_fichiers(dossier):
    return len([f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))])


# Verifications sur le nombre de fichiers, leurs types etc..
# Cree aussi le nom de l'archive qui sera créer
def verifications(folder1, folder2):
    
    # Est-ce que les dossiers existent bien ?
    if not os.path.isdir(folder1) or not os.path.isdir(folder2):
        print("Error: one or more specified folders do not exist.")
        sys.exit(1)
    
    # Est-ce qu'il y a le meme nombre d'images que de masques
    number1 = compter_fichiers(folder1)
    number2 = compter_fichiers(folder2)
    if number1 != number2:
        print("Error : ", number1, " images and ", number2, " masks\nExpected equals folder sizes.")
        sys.exit(1)
        
    return "u-net_dataset_" + str(number1) + ".zip"

def zip_folders(folder1, folder2, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in [folder1, folder2]:
            for root, _, files in os.walk(folder):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, os.path.dirname(folder))
                    zipf.write(abs_path, arcname=rel_path)
    print(f"Archive créée : {output_zip}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage : python build_u-net_dataset.py images_folder single_masks_folder")
        sys.exit(1)

    folder1 = sys.argv[1]
    folder2 = sys.argv[2]

    output = verifications(folder1, folder2)

    if(output):
        zip_folders(folder1, folder2, output)