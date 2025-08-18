"""
Usage : python gmm_processing.py <dataset_features.h5> [output_file] [model.pkl] [scaler.pkl]
(Leave model and scaler empty if you want to use the default latest trained one by Raphaël Kuhn)

Soon to be deprecated

"""

import sys, gmm_tools




if __name__ == "__main__":
    
    if len(sys.argv) < 4:
        print("Usage : python gmm_processing.py <dataset_features.h5> <model.joblib> <scaler.joblib> [output_csv_path]")
        print("(Leave model and scaler empty if you want to use the default latest trained one by Raphaël Kuhn)")
    elif len(sys.argv) == 4:
        gmm_tools.pipeline_process(sys.argv[1], sys.argv[2], sys.argv[3])    
    else:
        gmm_tools.pipeline_process(sys.argv[1], sys.argv[2], sys.argv[3], output_csv=sys.argv[4])