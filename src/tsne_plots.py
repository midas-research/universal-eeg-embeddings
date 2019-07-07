"""
tsne_plots.py: Creates TSNE plots for various embeddings
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
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
        mname = 0
        for model_name, Model in models:
            model = Model()
            print('<#######@@@@@@@#######>   Model   <#######@@@@@@@#######>', model_name)
            epoch = epochs[mname][dname]
            print('|=-=-=-=-=-=-=-=-=-     Dataset     -=-=-=-=-=-=-=-=-=|', dataset_name)
            print('Epoch Loaded {}'.format(epoch))
            """Evaluate saved the saved model(s)"""
            model.load_model_and_evaluate(loader_2, 2305, epoch=None)
            #exit(0)
            encoding, labels = model.get_encoding(loader_2)
            name = "./plots/tsne/{0}_{1}".format(dataset_name, model_name)
            np.save(name + "_encoding.npy", encoding)
            np.save(name + "_labels.npy", labels)
            print('|-*-*-*- saved labels -*-*-*-|')
            loader.tsneplot(name + "_with_labels", encoding, labels)
            mname += 1
        dname += 1
