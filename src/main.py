"""
main.py: Driver for all other tasks
"""

import argparse
# import models
# import loaders
import os
from models import CNN_GRU_Model, CNN_Only_Model, EEGNet_model, AutoEncoder_Model
from loaders import DataLoader
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNNC



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate or train the model(s) on the dataset(s)'
    )
    parser.add_argument('--data', type=str, default='SMR',
                        help='Dataset to use. SEED, SMR, BrainMNIST, '
                             'ThoughtViz or ALL')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain the model(s) on the dataset(s)')
    parser.add_argument('--model', type=str, default='all',
                        help='Model to use. CNN, CNN_GRU, EEG_Net, AE or All')
    parser.add_argument('--run_id', type=int, default=3,
                        help='run ID to use if retraining. Default 3')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs to use if retraining. Default 100')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    loader = DataLoader()
    datasets_dict = {
        # 'ERN': loader.get_ern,
        # 'SMR': lambda validation=False, subject=2: loader.get_smr(subject, validation),     # noqa
        # 'BMNIST': loader.get_bmnist11,
        # 'BMNIST_2': loader.get_bmnist2,
        # 'ThoughtViz': loader.get_thoughtviz,
        'ThoughtViz_char': loader.get_thoughtviz_char,
        'ThoughtViz_digit': loader.get_thoughtviz_digit,
        # 'SEED': loader.get_seed,
    }

    models_dict = {
        'CNN': CNN_Only_Model,
        'CNN_GRU': CNN_GRU_Model,
        'EEG_Net': EEGNet_model,
        'AE_rf': lambda: AutoEncoder_Model(RFC()),
        'AE_knn':lambda: AutoEncoder_Model(KNNC()),
    }

    if args.data == 'ALL':
        datasets = [[k, datasets_dict[k]] for k in datasets_dict]
    else:
        datasets = [[args.data, datasets_dict[args.data]]]

    if args.model == 'all':
        models = [[k, models_dict[k]] for k in models_dict]
    else:
        models = [[args.model, models_dict[args.model]]]
    for model_name, Model in models:
        model = Model()
        print('<#######@@@@@@@#######>   Model   <#######@@@@@@@#######>', model_name)
        for dataset_name, loader_2 in datasets:
            print('|=-=-=-=-=-=-=-=-=-     Dataset     -=-=-=-=-=-=-=-=-=|', dataset_name)
            if not args.retrain:
                """Evaluate saved the saved model(s)"""
                model.load_model_and_evaluate(loader_2, 1)
            else:
                """Train Model and print Evaluations"""
                model.train(loader_2, args.run_id, num_epochs=args.epochs)
