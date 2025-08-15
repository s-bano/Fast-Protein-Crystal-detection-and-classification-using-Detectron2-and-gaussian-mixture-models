import numpy as np
from sklearn.mixture import GaussianMixture
import gmm_tools


# === Paramètres ===
h5_path = "features_train_0812.h5"
n_clusters = 4  # à ajuster selon le nombre de classes supposées
output_model_path = "model_classif_0815.pkl"
pca_dim = gmm_tools.PCA_DIM


# === Étape 1: Charger tous les box_features ===
print("Extracting box features...")
all_box_features, image_names = gmm_tools.extract_h5(h5_path)


# === Étape 3: Normalisation ===
print("Data normalisation...")
X_scaled = gmm_tools.normalize(all_box_features, pca_dim=pca_dim)


# === Étape 4: Configuration du modele GMM a entrainer ===
gmm = GaussianMixture(
    n_components=n_clusters, 
    covariance_type='full', 
    reg_covar=1e-4, 
    random_state=42)


# === Étape 5: Entrainement du modele ===
gmm_infos = gmm_tools.training(X_scaled, gmm, output_model_path)


# === Étape 6: Analyse du model ===
gmm_model = gmm_infos['model']
ouptut_path = gmm_infos['model_path']
total_time = gmm_infos['training_time']
nbr_clusters_found = gmm_infos['nbr_clusters']


print(f"✅ GMM model trained and saved ({nbr_clusters_found} clusters, training_time: {total_time:.3f}s)")
print("Cluster weights:", gmm_model.weights_)
print("Means shape:", gmm_model.means_.shape)
print("Covariances shape:", gmm_model.covariances_.shape)
