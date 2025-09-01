import gmm_tools
import os, sys
import numpy as np
import pickle, random
import matplotlib.pyplot as plt

with open("all_info.pkl", "rb") as f:
    all_images_info = pickle.load(f)
    

print(gmm_tools.getStats(all_images_info))