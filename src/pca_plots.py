"""
pca_plots.py: Creates PCA plots for datasets
"""

# import argparse
# import models
# import loaders
import os
import numpy as np
from models import CNN_GRU_Model, CNN_Only_Model, EEGNet_model, AutoEncoder_Model
from loaders import DataLoader
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNNC



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    loader = DataLoader()
    datasets_dict = {
        'ERN': loader.get_ern,
        'SMR': lambda validation=False, subject=2: loader.get_smr(subject, validation),     # noqa
        'BMNIST': loader.get_bmnist11,
        'BMNIST_2': loader.get_bmnist2,
        'ThoughtViz': loader.get_thoughtviz,
        'SEED': loader.get_seed,
    }

    models_dict = {
        'CNN': CNN_Only_Model,
        'CNN_GRU': CNN_GRU_Model,
        'EEG_Net': EEGNet_model,
        'AE_rf': lambda: AutoEncoder_Model(RFC()),
        'AE_knn':lambda: AutoEncoder_Model(KNNC()),
    }

    datasets = [[k, datasets_dict[k]] for k in datasets_dict]

    models = [[k, models_dict[k]] for k in models_dict]

    epochs = np.array(
        [
            [54, 76, 25, 100, 70, 85],
            [70, 13, 74, 100, 98, 96],
            [87, 99, 95, 100, 99, 100],
            [100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100]
        ]
    )
    # Needs to be in the order of dictionary above for desired mapping

    dname = 0

    dname = 0
    for dataset_name, loader_2 in datasets:
        #if model_name not in ['AE_rf', 'AE_knn']:
        #    mname += 1
        #    continue
        print('<#######@@@@@@@#######>   Dataset   <#######@@@@@@@#######>', dataset_name)
        name = "plots/pca/" + dataset_name    
        loader.pcaplot(name, loader_2)
        dname += 1
