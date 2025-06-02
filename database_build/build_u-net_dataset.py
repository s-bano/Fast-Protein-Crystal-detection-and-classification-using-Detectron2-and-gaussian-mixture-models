import sys
import os
import zipfile

def get_matching_images(images_folder, masks_folder):
    mask_names = {os.path.splitext(f)[0] for f in os.listdir(masks_folder) if os.path.isfile(os.path.join(masks_folder, f))}
    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    return [f for f in image_files if os.path.splitext(f)[0] in mask_names]


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

def zip_folders(images_folder, masks_folder, output_zip):
    matching_images = get_matching_images(images_folder, masks_folder)
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in matching_images:
            abs_img = os.path.join(images_folder, file)
            zipf.write(abs_img, arcname=os.path.join('images', file))

            mask_file = file  # on suppose le même nom pour le masque
            abs_mask = os.path.join(masks_folder, mask_file)
            if os.path.exists(abs_mask):
                zipf.write(abs_mask, arcname=os.path.join('masks', mask_file))
    print(f"Archive créée : {output_zip}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : python build_u-net_dataset.py images_folder single_masks_folder")
        sys.exit(1)

    folder1 = sys.argv[1]
    folder2 = sys.argv[2]

    output = "u-net_dataset_" + str(compter_fichiers(folder2)) + ".zip"
    zip_folders(folder1, folder2, output)