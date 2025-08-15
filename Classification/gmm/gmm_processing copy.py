"""
Usage : python gmm_processing.py <dataset_features.h5> [output_file] [model.pkl] [scaler.pkl]
(Leave model and scaler empty if you want to use the default latest trained one by Raphaël Kuhn)
"""

import time, joblib, sys
import numpy as np
import gmm_tools
from sklearn.preprocessing import StandardScaler



def full_pipeline(h5_path, output_path="results_cluster.csv", model_path=gmm_tools.STANDARD_MODEL, scaler=StandardScaler(), pca_dim=gmm_tools.PCA_DIM):
    
    print("Openning file...")
    list_images_features, list_img_names = gmm_tools.extract_h5(h5_path)
    
    # Compter le nombre de cristaux par images avant la concatenation
    # Pour apres rerépartir les cristaux par images 
    print("Counting nbr cristals per images...")
    list_nbr_cristaux = [arr.shape[0] for arr in list_images_features]
    nbr_images = len(list_img_names)
    
    # Lancement du chronometre
    start_time = time.time()
        
    # Concatenation des features, pca et scaling
    X = gmm_tools.normalize(list_images_features, pca_dim, scaler)

    # Prediction des clusters
    print("Predicting clusters...")
    clusters = gmm_tools.classify_batch(X)
    print(clusters.shape)
    
    # Fin du chrono
    end_time = time.time()
    total_time = end_time - start_time
    
    resultat = gmm_tools.proportions(clusters)
    
    
    # RErepartir les cristaux par images
    count = 0
    list_clusters = []
    for N_i in list_nbr_cristaux:
        clusters_img = clusters[count:count+N_i]
        list_clusters.append(clusters_img)
        count = count+N_i

    # Enregistrement des clusters predits au format CSV
    print("Saving predictions...")
    gmm_tools.clusters2csv(list_img_names, list_clusters, output_path)
    
    # Affichage des statistiques
    print(f"\n✅ Clusters saved successfully: {output_path}")
    print(f"   Classification summary: \n{nbr_images} images processed in {total_time:.3f}s")
    print(f"Repartition: {resultat}") 
    
    image_to_clusters = dict(zip(list_img_names, list_clusters))
    
    return image_to_clusters



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
        scaler = joblib.load(sys.argv[4])
        full_pipeline(sys.argv[1], sys.argv[2], sys.argv[3], scaler)