import subprocess
import numpy as np
import glob
import os

if __name__ == '__main__':
#	loader = DataLoader()
    datasets = ['ERN','SMR','BMNIST','BMNIST_2', 'SEED', 'ThoughViz_image']
    ddirect = ['SEED_models', 'SMR_models', 'BMNIST_models', 'BMNIST_models', 'SEED_models', 'ThoughtViz_models']
    models = ['CNN', 'CNN_GRU', 'EEG_Net', 'AE_rf', 'AE_knn']
    mname_complete = ['CNN_Only', 'CNN_GRU', 'EEGNet', 'AE'] 
    epochs = np.array(
        [
            [19, 79, 31, 74, 93, 71],
            [50, 3, 40, 92, 68, 89],
            [17, 89, 82, 93, 94, 99],
            [98, 100, 38, 4, 100, 25]
        ]
    )


    mname = 0
    dname = 0
    run_id = 2305
    search_path = '/media/data_dump_1/group_2/data/'

    for mname_s in mname_complete:
        # if model_name not in ['AE_rf', 'AE_knn']:
        #     mname += 1
        #     continue
        # model = Model()
        dname = 0
        for dataset_name, spth in zip(datasets, ddirect):
            epoch = epochs[mname][dname]
            suffix = dataset_name
            # spth = dataset_name if dataset_name != 'ERN' else 'SEED'
            # spth += '_models'
            print('Epoch Loaded {}'.format(epoch))
            name = "{0}_{1}_{2}-model-improvement-{3:02d}*.h5".format(run_id, mname_s, suffix, epoch)
            fpath = os.path.join(search_path, spth, name)
            # model.load_model_and_evaluate(loader_2, 2305, epoch=epoch)
            # print(fpath)
            mdl = glob.glob(fpath)
            if len(mdl)==0:
                print(fpath)
            mpath = mdl[0]
            print(subprocess.run(['gdrive', 'upload', mpath, '-p', '1DjtEDQ-x-GbBdKnZ58mCFNw6xqWi0Tr-']))
            dname += 1
        mname += 1
