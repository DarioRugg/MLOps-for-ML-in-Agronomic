import os
import joblib
from os.path import join
import pickle
import clearml
import numpy as np
import pandas as pd
import time


os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

def get_data(dataset_name, dataset_id=None):
    if dataset_name == "splitted":
        strat_time = time.time()
        print("Downloading data...")
        strat_time = time.time()
        dataset_path = clearml.Dataset.get(**{"dataset_id": dataset_id} if dataset_id else {"dataset_name": "Dataset split", "dataset_project": "e-muse/DL_Wheat_with_clearml"}, alias="dataset_id").get_local_copy()
        print("Time to download data:", time.time() - strat_time)
        print(" - Dataset path:", dataset_path, end="\n\n")
        
        
        print("Reading HDF files...")
        X_train = pd.read_hdf(join(dataset_path, "x_train.h5"), key="data")
        X_test = pd.read_hdf(join(dataset_path, "x_test.h5"), key="data")
        print(" - Read in:", time.time() - strat_time, end="\n\n")
        
        print("Reading np arrays...")
        strat_time = time.time()
        y_train = np.load(join(dataset_path, "y_train.npy"))
        y_test = np.load(join(dataset_path, "y_test.npy"))        
        print(" - Read in:", time.time() - strat_time, end="\n\n")
        
        
        print("Loading Scaler...")
        strat_time = time.time()
        y_min_max_scaler = joblib.load(join(dataset_path, "y_min_max_scaler.pkl"))
        print(" - Loaded in:", time.time() - strat_time, end="\n\n")
    
        return X_train, y_train, X_test, y_test, y_min_max_scaler
    else:
        raise ValueError("The dataset name is not valid. Please provide a valid dataset name.")