"""
main.py: Driver for all other tasks
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
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    loader = DataLoader()
    datasets_dict = {
        'ERN': loader.get_ern,
        'SMR': lambda validation=False, subject=2: loader.get_smr(subject, validation),     # noqa
        'BMNIST': loader.get_bmnist11,
        'BMNIST_2': loader.get_bmnist2,
        'SEED': loader.get_seed,
        'ThoughtViz': loader.get_thoughtviz,
        'ThoughtViz_char': loader.get_thoughtviz_char,
        'ThoughtViz_digit': loader.get_thoughtviz_digit,
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
            [19, 79, 31, 74, 93, 71, 71, 71],
            [50, 3, 40, 92, 68, 89, 89, 89],
            [17, 89, 82, 93, 94, 99, 99, 99],
            [98, 100, 38, 4, 100, 25, 25, 25],
            [98, 100, 38, 4, 100, 25, 25, 25]
        ]
    )
    # Needs to be in the order of dictionary above for desired mapping

    mname = 0
    dname = 0

    for model_name, Model in models:
        # if model_name not in ['AE_rf', 'AE_knn']:
        #     mname += 1
        #     continue
        model = Model()
        print('<#######@@@@@@@#######>   Model   <#######@@@@@@@#######>', model_name)
        dname = 0
        for dataset_name, loader_2 in datasets:
            epoch = epochs[mname][dname]
            print('|=-=-=-=-=-=-=-=-=-     Dataset     -=-=-=-=-=-=-=-=-=|', dataset_name)
            print('Epoch Loaded {}'.format(epoch))
            """Evaluate saved the saved model(s)"""
            model.load_model_and_evaluate(loader_2, 2305, epoch=epoch)
            dname += 1
        mname += 1
