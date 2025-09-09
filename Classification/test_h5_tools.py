import h5_tools, os, sys
import numpy as np

h5_test_file = "all_features.h5"

def test_extract_h5_data():
    # Exemple d'utilisation
    filename = h5_test_file
    image_paths, box_features_list, boxes_list, global_features = h5_tools.extract_h5_data(filename)

    print(f"Found {len(image_paths)} images")
    print(f"Example: {image_paths[0]}, box_features shape: {box_features_list[0].shape}")
    


def test_split_h5():
    
    # Exemple d'utilisation
    h5_tools.split_h5_crystals(
        h5_test_file,
        "features_train_0909_tmp.h5",
        "features_val_0909_tmp.h5",
        train_ratio=0.7
    )
    
    os.remove("features_train_0909_tmp.h5")
    os.remove("features_val_0909_tmp.h5")
    

if __name__ == "__main__":
    try:
        test_extract_h5_data()
    except:
        sys.exit("Error: Test - test_extract_h5_data - FAILED")
    try:
        test_split_h5()
    except:
        sys.exit("Error: Test - test_split_h5 - FAILED")