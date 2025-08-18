#==========================================
#
# TRAIN AND USE A GMM CLASSIFICATION MODEL
#
#==========================================

import gmm_tools       

# ====== TRAINING =======

# METHOD 1: Simple no configuration method

h5_path = "features_train.h5"
gmm_model = gmm_tools.pipeline_training(h5_path)