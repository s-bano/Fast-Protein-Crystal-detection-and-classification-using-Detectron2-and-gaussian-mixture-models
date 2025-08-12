"""
Usage : python gmm_processing.py <dataset_features.h5> [output_file] [model.pkl] [scaler.pkl]
(Leave model and scaler empty if you want to use the default latest trained one by Raphaël Kuhn)
"""

import h5py, time, joblib, sys, csv
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from collections import Counter



STANDARD_MODEL = "model_classif_0812.pkl"
STANDARD_SCALER = "scaler_classif_0812.pkl"
PCA_DIM = 64

def proportions(lst):
    count = Counter(lst)
    total = len(lst)
    return {"Cluster id " + str(int(k)): v / total for k, v in count.items()}


# Classifier chaque cristal d'une image
def classify_image(image_features, model_path=STANDARD_MODEL, scaler_path=STANDARD_SCALER, pca=''):
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
    model = joblib.load(model_path)
    
    
    clusters = model.predict(image_features)
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


def full_pipeline(h5_path, output_path="results_cluster.csv", model_path=STANDARD_MODEL, scaler_path=STANDARD_SCALER):
    
    print("Openning file...")
    list_images_features, list_img_names = extract_h5(h5_path)
    
    # Compter le nombre de cristaux par images avant la concatenation
    # Pour apres rerépartir les cristaux par images 
    print("Counting nbr cristals per images...")
    list_nbr_cristaux = []
    for img_features in list_images_features:
        list_nbr_cristaux.append(img_features.shape[0])
    
    # LAncement du chronometre
    start_time = time.time()
    nbr_images = len(list_img_names)
        
    # Concatenation de sfeatures, pca et scaling
    print("Scaling and PCA features...")
    scaler = joblib.load(scaler_path)
    X = np.concatenate(list_images_features, axis=0)  # shape (total_boxes, D)
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=PCA_DIM)
    X = pca.fit_transform(X_scaled) 

    # Prediction des clusters
    print("Predicting clusters...")
    clusters = classify_image(X)
    print(clusters.shape)
    
    # Fin du chrono
    end_time = time.time()
    total_time = end_time - start_time
    
    resultat = proportions(clusters)
    
    
    # RErepartir les cristaux par images
    count = 0
    list_clusters = []
    for N_i in list_nbr_cristaux:
        clusters_img = clusters[count:count+N_i]
        list_clusters.append(clusters_img)
        count = count+N_i

    # Enregistrement des clusters predits au format CSV
    print("Saving predictions...")
    clusters2csv(list_img_names, list_clusters, output_path)
    
    # Affichage des statistiques
    print(f"\n✅ Clusters saved successfully: {output_path}")
    print(f"   Classification summary: \n{nbr_images} images processed in {total_time:.3f}s")
    print(f"Repartition: {resultat}") 

    
    sys.exit(0)


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