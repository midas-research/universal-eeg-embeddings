"""
grid_search.py: Do grid search on a model.
Currently implemented only for CNN_GRU on SMR
"""

import os
import argparse
# import models
# import loaders
import numpy as np
from models import CNN_GRU_Model, CNN_Only_Model, EEGNet_model
from loaders import DataLoader
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    param_grid = [
            {'num_gru': [2**i for i in range(5,7)],
             'pool1D': [False],
             'has2D': [False],
             'pool2D': [False],
             'poolAvg': [True]},
            {'num_gru': [2**i for i in range(5,7)],
             'pool1D': [True],
             'has2D': [False],
             'pool2D': [False],
             'poolAvg': [True, False]},
            {'num_gru': [32,150],#2**i for i in range(5,7)],
             'pool1D': [True, False],
             'has2D': [True],
             'pool2D': [True,False],
             'poolAvg': [True, False]}
        ]

    loader = DataLoader()
    list_splits = []
    X = []
    X_t = []
    Y = []
    Y_t = []
    train_size = 0
    for i in range(1):
        d = loader.get_smr(subject=i, return_idx=True) 
        data = d['data']
        X.append(data[0])
        X_t.append(data[2])
        Y.append(data[1])
        Y_t.append(data[3])
        list_splits.append(d['data_idx'])
        train_size+=data[0].shape[0]

    list_splits = [[i[0], i[1] + train_size] for i in list_splits]
    list_splits = [[np.append(i[0], i[1], axis=0), i[1]] for i in list_splits]

    X = np.concatenate(X, axis=0)
    X_t = np.concatenate(X_t, axis=0)
    X = np.append(X, X_t, axis=0)
    Y = np.concatenate(Y, axis=0)
    Y_t = np.concatenate(Y_t, axis=0)
    Y = np.append(Y, Y_t, axis=0)
    
    model = CNN_GRU_Model()
    model.suffix = 'SMR_GridSearch'
    model.num_classes = 4
    model.num_epochs=27
    #scorer = make_scorer
    clf = GridSearchCV(model, param_grid, cv = list_splits,
            scoring=None, verbose=4, refit=False)#, n_jobs=3, pre_dispatch='n_jobs')

    clf.fit(X, Y)
    exit(0)


    #TODO: Will do generalisation later
    parser = argparse.ArgumentParser(
        description='Grid Search a model on a dataset(s)'
    )
    parser.add_argument('--data', type=str, default='SMR',
                        help='Dataset to use. SEED, SMR, BrainMNIST, '
                             'ThoughtViz or ALL')
    #parser.add_argument('--retrain', action='store_true',
    #                    help='Retrain the model(s) on the dataset(s)')
    parser.add_argument('--model', type=str, default='CNN_GRU',
                        help='Model to use. CNN, CNN_GRU, EEG_Net, AE or All')
    parser.add_argument('--run_id', type=int, default=-1,
                        help='run ID to use if retraining. Default -1 (timestamp)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs to use if retraining. Default 100')
    args = parser.parse_args()
    loader = DataLoader()
    datasets_dict = {
        'SEED': loader.get_seed,
        'SMR': lambda validation=False, subject=2: loader.get_smr(subject, validation),     # noqa
        'BMNIST': loader.get_bmnist11,
        'ThoughtViz': loader.get_thoughtviz
    }

    models_dict = {
        'CNN': CNN_Only_Model,
        'CNN_GRU': CNN_GRU_Model,
        'EEG_Net': EEGNet_model
        # AE
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
        print('Model', model_name)
        for dataset_name, loader_2 in datasets:
            print('Dataset', dataset_name)
            if not args.retrain:
                """Evaluate saved the saved model(s)"""
                model.load_model_and_evaluate(loader_2, 1)
            else:
                """Train Model and print Evaluations"""
                model.train(loader_2, args.run_id, num_epochs=args.epochs)
