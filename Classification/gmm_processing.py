"""
Usage : python gmm_processing.py <dataset_features.h5> [output_file] [model.pkl] [scaler.pkl]
(Leave model and scaler empty if you want to use the default latest trained one by Raphaël Kuhn)
"""

import h5py, os, time, joblib, sys, csv
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


STANDARD_MODEL = "model_classif_0806.pkl"
STANDARD_SCALER = "scaler_classif_0806.pkl"


# Classifier chaque cristal d'une image
def classify_image(image_features, model_path=STANDARD_MODEL, scaler_path=STANDARD_SCALER):
    """
        This function will classify each detected cristals from it
        
        input:
            image_features: numpy array shape (N, D)
            -- model_path: Path to the model to be used to classify
            -- scaler_path: Path to the scaler to be used to normalize data before classify
            
        output:
            clusters: list(int) List of class_id predicted
    """
    if image_features.shape[0] == 0:
        return np.array([])
    
    # Charger le scaler et le modèle
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    
    X_scaled = scaler.fit_transform(image_features)
    clusters = model.predict(X_scaled)
    return clusters


def extract_h5(h5_path):
    """
    This function extracts the features from a h5 file
    
    outputs:
        - a list of features ready to be classified (list of arrays),
        - a list of corresponding image names.
    """
    all_box_features = []
    image_names = []
    with h5py.File(h5_path, "r") as f:
        for img_name in f.keys():
            box_feats = f[img_name]["box_features"][:]  # shape (N, D)
            all_box_features.append(box_feats)
            image_names.append(img_name)
    return all_box_features, image_names



# Enregistrer les clusters dans un csv
# NOTE: A changer pour combiner avec les csv fournis par le detecteur
def clusters2csv(image_names, clusters, output_csv):
    """
    Args:
        image_names: list of image names
        all_box_features: list of np.arrays (N_i, D) per image
        clusters_list_of_lists: list of lists of cristal indices per cluster
        output_csv: output csv filepath
    """

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        for img_name, cluster in zip(image_names, clusters):
            writer.writerow([img_name])
            n_cristaux = cluster.shape[0]
            for i in range(n_cristaux):                  
                cristal_name = f"cristal{i+1}"                  # A changer bientot pour combiner avec les csv de detection
                writer.writerow([cristal_name, cluster[i]])
                
    print(f"✅Clusters saved successfully: {output_csv}")


def full_pipeline(h5_path, output_path="results_cluster.csv", model_path=STANDARD_MODEL, scaler_path=STANDARD_SCALER):
    
    print("Openning file...")
    list_images_features, list_img_names = extract_h5(h5_path)
    
    print("Predicting clusters...")
    list_clusters = []
    start_time = time.time()
    nbr_images = 0
    for image in list_images_features:
        clusters = classify_image(image, model_path, scaler_path)
        list_clusters.append(clusters)
        nbr_images += 1
    end_time = time.time()
    total_time = end_time - start_time
    print(f"{nbr_images} images processed in {total_time:.3f}s")
    print("Saving predictions...")
    clusters2csv(list_img_names, list_clusters, output_path)


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage : python gmm_processing.py <dataset_features.h5> [output_file] [model.pkl] [scaler.pkl]")
        print("(Leave model and scaler empty if you want to use the default latest trained one by Raphaël Kuhn)")
    elif len(sys.argv) == 2:
        full_pipeline(sys.argv[1])
    elif len(sys.argv) == 3:
        full_pipeline(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        full_pipeline(sys.argv[1], sys.argv[2], sys.argv[3])    
    else:
        full_pipeline(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])