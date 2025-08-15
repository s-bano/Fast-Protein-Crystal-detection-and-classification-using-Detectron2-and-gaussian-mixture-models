import h5py, time, joblib, sys, csv, os
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

STANDARD_MODEL = "model_classif_0812.pkl"
STANDARD_SCALER = StandardScaler()
PCA_DIM = 64



# This function extracts the features from a h5 file
def extract_h5(h5_path):
    """
    This function extracts the features from a h5 file
    
    outputs:
        - a list of features (non concatenated and non-scaled) ready to be classified (list of arrays),
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


# Recapitulatif de la proportion de cristaux classifies dans chaque cluster
def proportions(lst):
    count = Counter(lst)
    total = len(lst)
    return {"Cluster id " + str(int(k)): v / total for k, v in count.items()}


# Conactene les box_features, normalize les donnees avec un scaler et applique une reudtcion PCA
def normalize(list_images_features, pca_dim=PCA_DIM, scaler=STANDARD_SCALER):
    
    X = np.concatenate(list_images_features, axis=0)  # shape (total_boxes, D)
    X_scaled = (scaler.fit_transform(X)).astype(np.float64)

    print("X_scaled shape:", X_scaled.shape)
    print("Nan ?", np.isnan(X_scaled).any())
    print("Inf ?", np.isinf(X_scaled).any())
    print("Max abs:", np.max(np.abs(X_scaled)))

    pca = PCA(n_components=pca_dim)
    X_scaled_pca = pca.fit_transform(X_scaled)
    
    return X_scaled_pca



# Train a GMM Classification model
def training(X_scaled, gmm_config, output_model_path="output_model.pkl"):
    
    print("GMM training...")
    start_time = time.time()

    gmm_config.fit(X_scaled)
    end_time = time.time()
    total_time = end_time - start_time
    nbr_clusters_found = (gmm_config.weights_ > 1e-3).sum()  

    # === Étape 5: Sauvegarde du modèle et du scaler ===
    print('Saving model...')
    joblib.dump(gmm_config, output_model_path)
    # joblib.dump(scaler, out_scaler_path)

    
    returns = {
        'model': gmm_config,
        'model_path': output_model_path,
        'training_time': total_time,
        'nbr_clusters': nbr_clusters_found,
    }
    
    return returns



# Classifier un batch concatener de features d'images (total_boxes, D)
def classify_batch(image_features, model_path=STANDARD_MODEL):
    """
        This function will classify each detected cristals from it
        
        input:
            image_features: numpy array shape (N, D)
            -- model_path: Path to the model to be used to classify
            
        output:
            clusters: list(int) List of class_id predicted
    """
    if image_features.shape[0] == 0:
        return np.array([])
    
    # Charger le scaler et le modèle
    model = joblib.load(model_path)
    clusters = model.predict(image_features)
    
    return clusters




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
                cristal_name = f"cristal{i+1}"             
                writer.writerow([cristal_name, cluster[i]])