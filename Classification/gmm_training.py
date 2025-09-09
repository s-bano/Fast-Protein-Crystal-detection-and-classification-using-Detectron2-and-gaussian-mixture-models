"""
python gmm_training.py features_train.h5
"""


import sys
import gmm.gmm_tools as gmm_tools
from pathlib import Path
import h5_tools
import numpy as np


def old_training():
    if len(sys.argv) < 2:
        print("Usage : python gmm_training.py features_train.h5")
    else:
        gmm_tools.pipeline_training(sys.argv[1])

    if len(sys.argv) < 2:
        print("Usage : python gmm_training.py features_train.h5")
        sys.exit()
        
image_paths, box_features_list, boxes_list, global_features = h5_tools.extract_h5_data(sys.argv[1])

print(len(box_features_list))    


all_box_features, _ = h5_tools.old_extract_h5("features_train_0813.h5")
print(all_box_features[0].shape)
print(box_features_list[0].shape)



gmm_tools.new_pipeline_training(box_features_list, model_path="model_classif_0909.joblib", scaler_path="scaler_classif_0909.joblib")

    
if __name__ == "__main__":
    
    box_features_list, image_paths = h5_tools.old_extract_h5("features_train_0813.h5")
    
    gmm_tools.pipeline_process(box_features_list, image_paths, "model_classif_0909.joblib", "scaler_classif_0909.joblib")
    
