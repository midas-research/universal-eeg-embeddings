from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import numpy as np
import os
import pandas as pd
import mne

class DataLoader(object):
    """
    This is the class that will helps loading data.
    It implements caching so second loads are constant time access
    in most cases
    """

    def __init__(self):
        self.prefix = os.getenv('GROUP2LOC', "/media/data_dump_1/group_2/data")
        self.seed = None
        self.bmnist = None
        self.thoughtviz = None
        self.smr = None
        self.ern = None

    def pcaplot(self, graph_name, loader_func):
        loader_dict = loader_func(validation=False)
        data = loader_dict['data']
        X = np.append(data[0], data[2], axis=0)
        print(X.shape)
        X = X.reshape(X.shape[0], -1)
        print(X.shape)
        Y = np.append(data[1], data[3], axis=0)
        Y = np.argmax(Y, axis=1)
        X_pca = PCA(n_components=3).fit_transform(X)
        np.save("{0} PCA 3D transform.npy".format(graph_name), X_pca)
        plt.scatter(X_pca[:,0], X_pca[:,1],
                c=Y, cmap="Pastel2")
        plt.title("{0} PCA Scatter Plot".format(graph_name))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("{0}_PCA_2D.jpg".format(graph_name))
        print("2D PCA for ", graph_name, "saved at", "{0}_PCA.jpg".format(graph_name))
        plt.clf()
        """ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(
                xs=X_pca[:,0],
                ys=X_pca[:,1],
                zs=X_pca[:,2],
                c=Y,
                cmap="Pastel2"
        )
        ax.set_xlabel('pca_one')
        ax.set_ylabel('pca_two')
        ax.set_zlabel('pca_three')
        plt.savefig("{0}_PCA_3D.jpg".format(graph_name))
        print("3D PCA for ", graph_name, "saved at", "{0}_PCA.jpg".format(graph_name))"""

    def tsneplot(self, graph_name, data, labels):
        """
        data - data to be plotted (num_samples, feature_dim)
        """
        data = np.array(data)
        X_embedded = TSNE(n_components=2).fit_transform(data)
        labels = np.argmax(labels, axis=1)
        np.save("{0} TSNE 2D.npy".format(graph_name), X_embedded)
        plt.scatter(X_embedded[:,0], X_embedded[:,1],
                c = labels, cmap="Pastel2")
        plt.title("{0} TSNE Scatter Plot".format(graph_name))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig("{0}_TSNE.jpg".format(graph_name))
        print("TSNE for ", graph_name, "saved at", "{0}_TSNE.jpg".format(graph_name))
        #plt.show()

    def topoplot(self, dataset, data):
        """
        dataset - SMR / SEED / ERN / BMNIST
        data - (num_channels,)
        num_channels - SMR: 22, SEED: 62, ERN:56, BMNIST: 4
        """

        info = pd.read_csv("Elecinfo.csv")
        info = np.array(info)
        d = {}
        for i in range(len(info)):
            row = info[i]
            d[row[0]] = [float(row[4]), float(row[5])]
        if dataset == "SMR":
            channels_SMR = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
            coords_SMR = []
            for ch in channels_SMR:
                coords_SMR.append(d[ch])
            mne.viz.plot_topomap(data, np.array(coords_SMR), names = channels_SMR, show_names = True)
        elif dataset == "SEED":
            channels_SEED = ["FP1", "FPz", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2"]
            coords_SEED = []
            for ch in channels_SEED:
                coords_SEED.append(d[ch])
            mne.viz.plot_topomap(data, np.array(coords_SEED), names = channels_SEED, show_names = True)
        elif dataset == "ERN":
            channels_ERN = ["FP1", "FP2", "AF7", "AF3", "AF4", "AF8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "POz", "PO8", "O1", "O2"]
            coords_ERN = []
            for ch in channels_ERN:
                coords_ERN.append(d[ch])
            mne.viz.plot_topomap(data, np.array(coords_ERN), names = channels_ERN, show_names = True)
        elif dataset == "BMNIST":
            channels_BMNIST = ["TP9", "FP1", "FP2", "TP10"]
            coords_BMNIST = []
            for ch in channels_BMNIST:
                coords_BMNIST.append(d[ch])
            mne.viz.plot_topomap(data, np.array(coords_BMNIST), names = channels_BMNIST, show_names = True)


    def get_full_path(self, pth):
        """
        Helper function to improve portability to certain extent
        by segregating task of dataset location identification to
        self.prefix
        Thus when ported to colab or another machine, only self.prefix
        must be changed provided `data/` folder keeps the same structure.
        """
        return os.path.join(self.prefix, pth)

    def get_partitions(self, data, labels, validation=False):
        train_idx, test_idx = train_test_split(np.arange(data.shape[0]), test_size=0.25, random_state=42)   # noqa
        if validation is True:
            train_idx, valid_idx = train_test_split(train_idx, test_size=0.25, random_state=26)             # noqa
        train_x = data[train_idx]
        train_y = labels[train_idx]

        test_x = data[test_idx]
        test_y = labels[test_idx]

        if validation is True:
            valid_x = data[valid_idx]
            valid_y = labels[valid_idx]
            return [train_x, train_y, test_x, test_y, valid_x, valid_y]

        return [train_x, train_y, test_x, test_y, test_x, test_y]

    def get_seed(self, validation=False):
        """
        This will load SEED data as train and test.
        The task is ternary classification
        """
        if self.seed is None:
            with open(self.get_full_path('SEED_62/SEED_data_62.pk'), 'rb') as f:    # noqa
                data = pickle.load(f)
            with open(self.get_full_path('SEED_62/SEED_labels_62.pk'), 'rb') as f:   # noqa
                labels = pickle.load(f)
            data = data.reshape(data.shape + (1,))
            self.seed = (data, labels)
        else:
            (data, labels) = self.seed

        lvals = np.unique(labels)
        one_hot_labels = (labels == lvals[:, np.newaxis]).T

        retval = {
            'num_classes': 3,
            'data': self.get_partitions(data, one_hot_labels, validation=validation), # noqa
            'name': 'SEED',
            'model_save_dir': self.get_full_path('SEED_models')
        }

        return retval

    def get_bmnist11(self, validation=False, binarize=False):
        """
        Returns the 11 class version for BMNIST dataset.
        10 digits and 1 for not others.
        """
        if self.bmnist is None:
            data = np.load(self.get_full_path('BMNIST_4/bmnist_x.npy'))
            data = data.reshape(data.shape + (1,))
            labels = np.load(self.get_full_path('BMNIST_4/bmnist_y.npy'))
            labels = labels.flatten()
            self.bmnist = (data, labels)
        else:
            (data, labels) = self.bmnist

        if binarize is True:
            change_indexes = (labels != -1)
            labels = labels.copy()
            labels[change_indexes] = 1

        lvals = np.unique(labels)
        one_hot_labels = (labels == lvals[:, np.newaxis]).T

        retval = {
            'num_classes': 11 if not binarize else 2,
            'data': self.get_partitions(data, one_hot_labels, validation=validation), # noqa
            'name': 'BMNIST' if not binarize else 'BMNIST_2',
            'model_save_dir': self.get_full_path('BMNIST_models')
        }

        return retval

    def get_bmnist2(self, validation=False):
        """
        Return the 2 class version for BMNIST.
        Looking or not Looking
        """
        return self.get_bmnist11(validation=validation, binarize=True)

    def get_smr(self, subject=None, validation=False, return_idx=False):
        """
        Returns within subject SMR data
        Subject = The subject for which to return data
        """
        assert subject is not None
        assert subject < 9 and subject >= 0
        # assert validation is False              # noqa, No validation data available for any subject
        # use test data for the same

        if self.smr is None:
            train_x = np.load(self.get_full_path('SMR/X_train.npy'))
            train_x = train_x.reshape((-1,) + train_x.shape[2:] + (1,))             # noqa, important to add 1 so as to match the format
            train_y = np.load(self.get_full_path('SMR/Y_train.npy'))
            train_y = train_y.flatten()

            lvals = np.unique(train_y)
            train_y = (train_y == lvals[:, np.newaxis]).T

            test_x = np.load(self.get_full_path('SMR/X_test.npy'))
            test_x = test_x.reshape((-1,) + test_x.shape[2:] + (1,))
            test_y = np.load(self.get_full_path('SMR/Y_test.npy'))
            test_y = test_y.flatten()

            test_y = (test_y == lvals[:, np.newaxis]).T

            train_data = (train_x, train_y)
            test_data = (test_x, test_y)
            self.smr = (train_data, test_data)
        else:
            ((train_x, train_y), (test_x, test_y)) = self.smr

        # get the apt subject data from training set
        num_samples = train_x.shape[0] // 9
        train_idx = np.arange(num_samples * subject, num_samples * (subject + 1))
        train_feats = (train_x[train_idx])  # noqa
        train_labels = (train_y[train_idx])  # noqa

        num_samples = test_x.shape[0] // 9
        test_idx = np.arange(num_samples * subject, num_samples * (subject + 1))
        test_feats = (test_x[test_idx])  # noqa
        test_labels = (test_y[test_idx])  # noqa

        retval = {
            'num_classes': 4,
            'data': [train_feats, train_labels, test_feats, test_labels, test_feats, test_labels], # noqa
            'name': 'SMR',
            'model_save_dir': self.get_full_path('SMR_models')
        }
        if (return_idx):
            retval['data_idx'] = [train_idx, test_idx]

        return retval

    def get_thoughtviz(self, validation=False, tp='image'):
        """
        Load data for ThoughtViz. Supports all three EEG datasets, i.e
        Image, Char, Digit
        10 class classification
        """
        assert tp in ['image', 'char', 'digit']
        # assert validation is False              # noqa, No validation data for ThoughtViz
        # Using test data for the same

        if self.thoughtviz is None:
            self.thoughtviz = {}

        if tp not in self.thoughtviz:
            # Dynamically cache to save time next time
            whole_data = pickle.load(open(self.get_full_path('thoughtviz/{}/data.pkl'.format(tp)), 'rb'), encoding='bytes')    # noqa
            self.thoughtviz[tp] = whole_data
        else:
            whole_data = self.thoughtviz[tp]

        train_x, train_y = whole_data[b'x_train'], whole_data[b'y_train']
        test_x, test_y = whole_data[b'x_test'], whole_data[b'y_test']

        retval = {
            'num_classes': 10,
            'data': [train_x, train_y, test_x, test_y, test_x, test_y],
            'name': 'ThoughViz_{}'.format(tp),
            'model_save_dir': self.get_full_path('ThoughtViz_models')
        }

        return retval

    def get_thoughtviz_image(self, validation=False):
        """
        ThoughViz images EEG Data
        """
        return self.get_thoughtviz(validation=validation, tp='image')

    def get_thoughtviz_char(self, validation=False):
        """
        ThoughViz Characters EEG Data
        """
        return self.get_thoughtviz(validation=validation, tp='char')

    def get_thoughtviz_digit(self, validation=False):
        """
        ThoughViz Digits EEG Data
        """
        return self.get_thoughtviz(validation=validation, tp='digit')

    def get_ern(self, validation=False):
        """
        Return ERN dataset from Kaggle
        """
        if self.ern is None:
            with open(self.get_full_path('ERN/ERN_data.pickle'), 'rb') as f:    # noqa
                data = pickle.load(f)
            with open(self.get_full_path('ERN/ERN_labels.pickle'), 'rb') as f:   # noqa
                labels = pickle.load(f)
            data = data.reshape(data.shape + (1,))
            data = np.swapaxes(data, 1, 2)
            self.ern = (data, labels)
        else:
            (data, labels) = self.ern

        lvals = np.unique(labels)
        one_hot_labels = (labels == lvals[:, np.newaxis]).T

        retval = {
            'num_classes': len(lvals),
            'data': self.get_partitions(data, one_hot_labels, validation=validation), # noqa
            'name': 'ERN',
            'model_save_dir': self.get_full_path('SEED_models')
        }

        return retval



if __name__ == '__main__':
    # dl = DataLoader()

    # d0 = dl.get_smr(subject=8)
    # print('===>', len(d0['data']), d0['data'][0].shape, d0['data'][1].shape)

    # # d1 = dl.get_seed()
    # # print('===>', len(d1), d1['data'][0].shape, d1['data'][1].shape)

    # # d2 = dl.get_bmnist11()
    # # print('===>', len(d2), d2['data'][0].shape, d2['data'][1].shape)

    # # d3 = dl.get_bmnist2()
    # # print('===>', len(d3), d3['data'][0].shape, d3['data'][1].shape)

    # # d4 = dl.get_thoughtviz_char()
    # # print('===>', len(d4), d4['data'][0].shape, d4['data'][1].shape)

    # d5 = dl.get_ern()
    # print('===>', len(d5), d5['data'][0].shape, d5['data'][1].shape)

    loader = DataLoader()

    datasets_dict = {
        'ERN': loader.get_ern,
        'SMR': lambda validation=False, subject=2: loader.get_smr(subject, validation),     # noqa
        'BMNIST': loader.get_bmnist11,
        # 'BMNIST_2': loader.get_bmnist2,
        'SEED': loader.get_seed,
        # 'ThoughtViz': loader.get_thoughtviz,
    }

    datasets = [[k, datasets_dict[k]] for k in datasets_dict]

    k = 'A'

    for dname, ldr in datasets:
        fig, axes = plt.subplots(1, 1)
        p = ldr(validation=True)
        data = p['data'][0][-1,:,0].reshape(-1,)
        loader.topoplot(p['name'], data)
        axes.set_title('({}) Topoplot for {}'.format(k, dname))
        fig.savefig('./topoplots_new/{}_topoplot.png'.format(dname))
        k = chr(ord(k)+1)
        print('Done {}, {}'.format(dname, p['name']))
