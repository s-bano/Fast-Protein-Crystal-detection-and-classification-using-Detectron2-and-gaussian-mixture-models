'''
This script aims to classify cistals into shapes A, B, C starting from masks
'''

from PIL import Image
import numpy as np
import sys
import os


# This function simply count the number of active pixels of a mask
def count_pixels(mask_path):
    img = Image.open(mask_path).convert('RGB')
    arr = np.array(img)
    non_black_pixels = np.sum(np.any(arr != 0, axis=2))
    return non_black_pixels


# The goal of this function is to starting from the aire of a shape, classify it into spe A, B, C
def aire2shape(aire):
    return



if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python edge2shape.py <mask_image> <scale>\nInfo: 1 pixel = <scale>mm")
        sys.exit(1)
    

    nbr_red = count_pixels(sys.argv[1])
    print("Number of non-black pixels: ", nbr_red)
    aire = nbr_red * float(sys.argv[2])
    print("Aire totale: ", aire)

