"""
python gmm_training.py features_train.h5
"""


import sys
import gmm_tools


    
    
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage : python gmm_training.py features_train.h5 [output_model_path]")
    elif len(sys.argv) < 3:
        gmm_tools.pipeline_training(sys.argv[1])
    else:
        gmm_tools.pipeline_training(sys.argv[1], model_path=sys.argv[2])
    
