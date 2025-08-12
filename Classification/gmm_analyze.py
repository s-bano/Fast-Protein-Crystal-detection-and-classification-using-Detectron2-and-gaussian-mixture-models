import h5py, os, time, joblib, sys, csv
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from collections import Counter

model = joblib.load("model_classif_0808.pkl")
print("Cluster weights:", model.weights_)
print("Means shape:", model.means_.shape)
print("Covariances shape:", model.covariances_.shape)