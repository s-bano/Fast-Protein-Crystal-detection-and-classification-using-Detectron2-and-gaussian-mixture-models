'''
This script allosw the user to run the detection of crystals on images
'''
import os
import sys
from utils import *
from PIL import Image
import numpy as np
import argparse
from pathlib import Path



def detect_folder(input_dir):
    '''
    Cette fonction cree un dossier result et met dedans les images du dossier d'entree
    avec des carres verts autour des cristaux
    '''
    output_dir = 'results3'

    pipe = Pipeline(detection_model_path=DETECT_MODEL, segmentation_model_path=SEGMENT_MODEL)

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):  # Filtre images
                filepath = os.path.join(root, filename)
                print(f"Processing {filepath}...")

                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                pipe.generate_detections(filepath, output_subdir)
                pipe.export_detections_to_csv()
                out_image, out_list = pipe.display_detections()
                
                output_path = os.path.join(output_subdir, f"{os.path.splitext(filename)[0]}_detected.png")
                plt.imsave(output_path, out_image)
                
                
def extract_images(input_dir, output_dir):

    pipe = Pipeline(detection_model_path=DETECT_MODEL, segmentation_model_path=SEGMENT_MODEL)
    
    processed = 0
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):  # Filtre images
                filepath = os.path.join(root, filename)
                print(f"Processing {filepath}...")
                
                # Detection des cristaux
                pipe.generate_detections(filepath, 'results')
                _, out_list = pipe.display_detections()
                display_list = list(map(lambda x : cv2.resize(x, (200, 200)),out_list))
                
                # Sauvegarde des images
                for idx, obj in enumerate(display_list):
                    newfilename = Path(filename).stem + f"_{idx:02d}.jpg"
                    save_path = output_dir + "/" + newfilename
                    print(f"Saving cristal in {save_path}...")
                    plt.imsave(save_path, obj)
                    
                print(f"{processed+1} images processed")

                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exemple avec -a et -b")

    # Définition des options avec un paramètre obligatoire
    parser.add_argument("-a", metavar="CONFIG", help="Detection seule des cristaux sur l'image")
    parser.add_argument("-b", nargs="+", help="Extraction d'images a part pour chaque cristal")

    args = parser.parse_args()

    if args.a:
        print(f"Option -a détectée avec valeur : {args.a}")
        input_path = args.a
        if not os.path.isdir(input_path):
            print("Usage: python run_detection -a <INPUT_FOLDER>")
            sys.exit(1)
        else:
            detect_folder(input_path)
    elif args.b:
        print(f"Option -b détectée avec valeur : {args.b}")
        input_path = args.b[0]
        if not os.path.isdir(input_path):
            print("Usage: python run_detection -b <INPUT_FOLDER> <OUTPUT_FOLDER>")
            sys.exit(1)
        else:
            output_dir = args.b[1]
            extract_images(input_path, output_dir)
    else:
        print("Usage: python run_detection -a/-b")
