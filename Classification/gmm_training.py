import h5py, os, time, joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# === Paramètres ===
h5_path = "features_train_0806.h5"
n_clusters = 3  # à ajuster selon le nombre de classes supposées
pca_dim = 64
output_model_path = "model_classif_0808.pkl"
output_scaler_path = "scaler_classif_0808.pkl"
normalize = True


# === Étape 1: Charger tous les box_features ===
print("Extracting box features...")
all_box_features = []

with h5py.File(h5_path, "r") as f:
    nbr_images = len(f.keys())
    for img_name in f.keys():
        box_feats = f[img_name]["box_features"][:]  # shape (N, D)
        all_box_features.append(box_feats)




# === Étape 2: Concaténation ===
print("Concatenate box features...")
X = np.concatenate(all_box_features, axis=0)  # shape (total_boxes, D)


# === Étape 3: Normalisation (optionnelle mais recommandée) ===
print("Data normalisation...")
if normalize:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X
X_scaled = X_scaled.astype(np.float64)

print("X_scaled shape:", X_scaled.shape)
print("Nan ?", np.isnan(X_scaled).any())
print("Inf ?", np.isinf(X_scaled).any())
print("Max abs:", np.max(np.abs(X_scaled)))

pca = PCA(n_components=pca_dim)
X_scaled = pca.fit_transform(X_scaled)
print(X.shape)


# === Étape 4: Entraînement du GMM ===
print("GMM training...")
start_time = time.time()

gmm = GaussianMixture(
    n_components=n_clusters, 
    covariance_type='full', 
    reg_covar=1e-4,  # augmente si le problème persiste
    random_state=42)

gmm.fit(X_scaled)
end_time = time.time()
total_time = end_time - start_time

# === Étape 5: Sauvegarde du modèle et du scaler ===
print('Saving model...')
joblib.dump(gmm, output_model_path)
joblib.dump(scaler, output_scaler_path)

print(gmm.weights_)
print(f"✅ GMM model trained and saved ({n_clusters} clusters, {X.shape[0]} cristaux, training_time: {total_time:.3f}s)")