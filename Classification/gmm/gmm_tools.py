import h5py, time, joblib, csv, os, sys
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline


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
    return {
        "Cluster id " + str(int(k)): round((v / total) * 100, 2)
        for k, v in count.items()
    }


# Train a GMM Classification model
def training(X_scaled, gmm_config, output_model_path="output_model.pkl"):
    
    print("GMM training...")
    start_time = time.time()

    gmm_config.fit(X_scaled)
    end_time = time.time()
    total_time = end_time - start_time
    nbr_clusters_found = (gmm_config.weights_ > 1e-3).sum()  
    
    returns = {
        'model': gmm_config,
        'model_path': output_model_path,
        'training_time': total_time,
        'nbr_clusters': nbr_clusters_found,
    }
    
    return returns


def pipeline_training(h5_path, **kwargs):
    """
        This function will perform a full pipeline training without any configuration
        Usefull if you don't know what to do or for a first model
        For advanced configured training, please check the tuto on the github page
        
        input:
            h5_path: Path to the images features to train on
            -- model_path: Path to the where the model should be saved
            -- n_clusters: Number of clusters to find approximately
            -- pca_dim: Number of dimensions the features will be reduct (PCA) (must be the same bteween training and processing)
            -- sacler: The scaler used to normalize data before treatment (default: StandardScaler)
            
        output:
            gmm_model: the gmm model trained
    """
    
    output_model_path = kwargs.get("model_path", "model_classif.joblib")
    output_scaler_path = kwargs.get("scaler_output", "scaler_classif.joblib")
    n_clusters = kwargs.get("nbr_clusters", 4)
    pca_dim = kwargs.get("pca_dim", PCA_DIM)
    scaler = kwargs.get("scaler", STANDARD_SCALER)

    # === Étape 1: Charger tous les box_features ===
    print("Extracting box features...")
    all_box_features, _ = extract_h5(h5_path)

    # === Étape 2: Normalisation ===
    print("Data normalisation...")
    X = np.concatenate(all_box_features, axis=0)
    pipe = Pipeline([
        ("scaler", scaler),
        ("pca", PCA(n_components=pca_dim, svd_solver="auto", random_state=0)),
    ])
    X_scaled = pipe.fit_transform(X)


    # === Étape 3: Configuration du modele GMM a entrainer ===
    gmm = GaussianMixture(
        n_components=n_clusters, 
        covariance_type='full', 
        reg_covar=1e-4, 
        random_state=42
    )


    # === Étape 4: Entrainement du modele ===
    gmm_infos = training(X_scaled, gmm, output_model_path)


    # === Étape 5: Analyse du model ===
    gmm_model = gmm_infos['model']
    ouptut_path = gmm_infos['model_path']
    total_time = gmm_infos['training_time']
    nbr_clusters_found = gmm_infos['nbr_clusters']
    
    
    # === Étape 6: Sauvegarde du modèle et du scaler ===
    print('Saving model and scaler...')
    joblib.dump(gmm_model, output_model_path)
    joblib.dump(pipe, output_scaler_path)


    # === Étape 7: Prints finaux d'infos utiles ===
    print(f"✅ GMM model trained and saved here: {ouptut_path}")
    print(f"✅ GMM scaler trained and saved here: {output_scaler_path}")
    print(f"{nbr_clusters_found} clusters found in training_time {total_time:.3f}s)")
    print("Cluster weights:", gmm_model.weights_)
    print("Means shape:", gmm_model.means_.shape)
    print("Covariances shape:", gmm_model.covariances_.shape)
    
    return gmm_model


# Classifier un batch de features d'images (total_boxes, D)
def classify_batch(features, model, scaler):
    """
        This function will classify each detected cristals from it
        
        input:
            features: numpy array shape (N, D)
            -- model_path: Path to the model to be used to classify
            -- scaler_path: Path to the scaler to be used to normalize data beforehand
            
        output:
            clusters: list(int) List of class_id predicted
    """
    if features.shape[0] == 0:
        return np.array([])
    elif features.ndim == 1:
        features = features.reshape(1, -1)  # devient (1, 1024)
    
    # Normalize data
    X_scaled = scaler.transform(features)
    
    # Predict clusters
    clusters = model.predict(X_scaled)
    
    return clusters


# Enregistrer les clusters dans un csv
# NOTE: DEPRECATED
#       See crystalmetrics.export instead
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


# full pipeline to classify images fetaures from a h5 file
# NOTE: DEPRECATED
#       See the colab notebook Detection&Classification instead             
def pipeline_process(h5_path, model_path, scaler_path, **kwargs):
    
    output_csv = kwargs.get("output_csv", "results_classif.csv")
    
    print("Openning file...")
    all_box_features, image_names = extract_h5(h5_path)
    
    list_nbr_cristaux = [arr.shape[0] for arr in all_box_features]
    nbr_images = len(image_names)
    
    X = np.concatenate(all_box_features, axis=0)
    
    start_time = time.time() 
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    clusters = classify_batch(X, model, scaler)
    total_time = time.time() - start_time
    
    # RErepartir les cristaux par images
    count = 0
    list_clusters = []
    for N_i in list_nbr_cristaux:
        clusters_img = clusters[count:count+N_i]
        list_clusters.append(clusters_img)
        count = count+N_i
        
    # Enregistrement des clusters predits au format CSV
    print("Saving predictions...")
    clusters2csv(image_names, list_clusters, output_csv)
    
    # Affichage des statistiques
    print(f"\n✅ Clusters saved successfully: {output_csv}")
    print(f"   Classification summary: \n{nbr_images} images processed in {total_time:.3f}s")
    print(f"Repartition: {proportions(clusters)}") 
    
    image_to_clusters = dict(zip(image_names, list_clusters))
    
    return image_to_clusters
        
    
# Conactene les box_features, normalize les donnees avec un scaler et applique une reudtcion PCA
# NOTE: DEPRECATED
#       Doesn't need to be use anymore and will likely fail
def _normalize(list_images_features, pca_dim=PCA_DIM, scaler=STANDARD_SCALER):
    
    X = np.concatenate(list_images_features, axis=0)  # shape (total_boxes, D)
    X_scaled = (scaler.fit_transform(X)).astype(np.float64)

    print("X_scaled shape:", X_scaled.shape)
    print("Nan ?", np.isnan(X_scaled).any())
    print("Inf ?", np.isinf(X_scaled).any())
    print("Max abs:", np.max(np.abs(X_scaled)))

    pca = PCA(n_components=pca_dim)
    X_scaled_pca = pca.fit_transform(X_scaled)
    
    return X_scaled_pca
