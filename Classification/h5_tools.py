import h5py
import numpy as np
from sklearn.model_selection import train_test_split




def extract_h5_data(filename):
    """
    Extract all image data from an HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.

    Returns
    -------
    image_paths : list of str
        Full paths of images (HDF5 groups containing datasets).
    box_features_list : list of np.ndarray
        List of box_features arrays, one per image.
    boxes_list : list of np.ndarray
        List of boxes arrays, one per image.
    global_features_list : list of np.ndarray
        List of global_features arrays, one per image.
    """
    image_paths = []
    box_features_list = []
    boxes_list = []
    global_features_list = []

    def visit_group(group, path=""):
        for key in group.keys():
            item = group[key]
            current_path = f"{path}/{key}" if path else key
            if isinstance(item, h5py.Group):
                # Vérifie si c'est un groupe d'image final
                if all(ds in item for ds in ["box_features", "boxes", "global_features"]):
                    image_paths.append(current_path)
                    # Met None si dataset inexistant
                    box_features_list.append(item["box_features"][:] if "box_features" in item else None)
                    boxes_list.append(item["boxes"][:] if "boxes" in item else None)
                    global_features_list.append(item["global_features"][:] if "global_features" in item else None)
                else:
                    visit_group(item, current_path)

    with h5py.File(filename, "r") as f:
        visit_group(f)

    return image_paths, box_features_list, boxes_list, global_features_list




def split_h5_crystals(input_h5, output_train_h5, output_val_h5, train_ratio=0.7):
    """
    Split crystals from an HDF5 file into training and validation sets.

    Parameters
    ----------
    input_h5 : str
        Path to the original HDF5 file.
    output_train_h5 : str
        Path to save training HDF5.
    output_val_h5 : str
        Path to save validation HDF5.
    train_ratio : float
        Fraction of crystals to put in training set.
    """
    # Extraction
    image_paths, box_features_list, boxes_list, global_features_list = extract_h5_data(input_h5)

    # Concaténer tous les cristaux ensemble
    all_box_features = np.concatenate([bf for bf in box_features_list if bf is not None and len(bf) > 0], axis=0)
    all_boxes = np.concatenate([b for b in boxes_list if b is not None and len(b) > 0], axis=0)
    # Pour global_features, répéter la feature globale par nombre de cristaux
    all_global_features = np.concatenate([
        np.repeat(gf[np.newaxis, :], bf.shape[0], axis=0)
        for bf, gf in zip(box_features_list, global_features_list)
        if bf is not None and len(bf) > 0
    ], axis=0)

    # Split
    idx_train, idx_val = train_test_split(np.arange(all_box_features.shape[0]), train_size=train_ratio, shuffle=True, random_state=42)

    # Fonction pour créer un HDF5
    def save_h5(filename, indices):
        with h5py.File(filename, "w") as f:
            f.create_dataset("box_features", data=all_box_features[indices])
            f.create_dataset("boxes", data=all_boxes[indices])
            f.create_dataset("global_features", data=all_global_features[indices])

    save_h5(output_train_h5, idx_train)
    save_h5(output_val_h5, idx_val)

    print(f"Saved {len(idx_train)} crystals to {output_train_h5}")
    print(f"Saved {len(idx_val)} crystals to {output_val_h5}")





# NOTE: DEPRECATED
# This function extracts the features from a h5 file OLD
def old_extract_h5(h5_path):
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