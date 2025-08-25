import gmm_tools
import os, sys
import numpy as np
import pickle, random
import matplotlib.pyplot as plt

    
    
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
    
    
def test_imageInfo2arr(all_images_info):
    image_info = all_images_info[0]
    arr = gmm_tools.image_info_to_arr(image_info)
    print(arr)
    

def test_concat_horizontal(all_images_info):
    
    arrays = []
    for i in range(4):
        arrays.append(all_images_info[i]["crystal_info"])
        
    try:
        arr = gmm_tools.concat_horizontal(arrays)
        if isinstance(arr[0][1], str) or isinstance(arr[0][2], str):
            print("FAIL: Variable is a string (str), but a float was expected.")
            return False
    except:
        print("FAIL: Concatenation didn't work")
        return False
    
    print("SUCCESS: Concatenation by listoflist working")
    return True


root_dir = '/Users/pagatok/Projets/Stage/database_build/crystal_images_filip'


with open("all_info.pkl", "rb") as f:
    all_images_info = pickle.load(f)


gmm_tools.filip_save(all_images_info, root_dir)
#test_imageInfo2arr(all_images_info)
#get_example(all_images_info)
#test_output_images(all_images_info)
#test_concat_horizontal(all_images_info)
