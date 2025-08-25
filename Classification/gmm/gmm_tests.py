import gmm_tools
import os, sys, random
import numpy as np
from collections import defaultdict
from pathlib import Path
from openpyxl import Workbook
import pickle
import matplotlib.pyplot as plt


with open("all_info.pkl", "rb") as f:
    all_images_info = pickle.load(f)
    
    
def test_output_images(all_images_info):
    
    list_failed_images = []
    x = random.randint(0, len(all_images_info)-1)  # entier entre 0 et N-1 inclus
    
    for i, image_info in enumerate(all_images_info):
        try:
            _ = plt.imshow(image_info["image"])  # tente de charger
        except Exception:
            list_failed_images.append(image_info["name"])
        if i == x:
            plt.imshow(image_info["image"])
            plt.axis("off")
            plt.show()
    
    if len(list_failed_images) == 0:
        print("SUCCESS: no corrupted images found")
        return True
    else:
        print(f"FAIL: {len(list_failed_images)} images are corrupted.")
        return False


def get_example(all_images_info):
    image_info = all_images_info[0]
    
    print(image_info["name"])
    crystal_arr = np.array(image_info['crystal_info'])
    print(crystal_arr.shape)


root_dir = '/Users/pagatok/Projets/Stage/database_build/crystal_images_filip'
gmm_tools.filip_save(all_images_info, root_dir)



#get_example(all_images_info)
#test_output_images(all_images_info)
    
