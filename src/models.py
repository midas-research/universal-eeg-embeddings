"""
Models.py: Stores architectures for all models trained
keras backend assumes 'channels_last' by default
"""
import os
import glob
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.layers import SpatialDropout2D, AveragePooling2D, Input, Dropout
from keras.layers import DepthwiseConv2D, SeparableConv2D, Activation
from keras.layers import GRU, TimeDistributed, LSTM, Reshape
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.constraints import max_norm

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,\
    precision_score, roc_auc_score, recall_score

import numpy as np
import glob
import datetime


class BaseModel(BaseEstimator):
    """Base class for a model."""

    def __init__(self, model_name, num_classes=4):
        self.name = model_name
        self.model = Sequential()
        self.classifier = None
        self.num_classes = num_classes
        self.param_suffix = ''
        # TODO: Make self.suffix using class params if that is also ''
        # Will be done after integrating params into base class
        self.suffix = ''

    def model_init(self, channels, observations, num_classes):
        """
        The model will be defined in this function
        """
        pass

    def get_encoding(self, loader_func):
        if(not hasattr(self, 'encoder') or self.encoder == None):
            self.encoder = Model(inputs=self.model.input, outputs=self.model.layers[self.enc_layer_ind].output)
        loader_dict = loader_func(validation=False)
        data = loader_dict['data']
        self.num_classes = loader_dict['num_classes']
        model_save_dir = loader_dict['model_save_dir']
        suffix = loader_dict['name']
        return self.encoder.predict(np.append(data[0], data[2], axis=0)), np.append(data[1], data[3], axis=0)

    def fit(self, X, Y, val_data=None, num_epochs=35, batch_size=64, 
            model_save_dir='test_models', run_id=-1, suffix=''):
        """Fit function, fits the models and saves models
        using config as passed in arguments"""
        
        channels = X.shape[1]
        observations = X.shape[2]
        self.model_init(channels, observations, self.num_classes)
        if (val_data==None):
            X, X_val, Y, Y_val = train_test_split(X,Y,test_size=0.25)
            val_data = [X_val, Y_val]
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        if (suffix==''):
            suffix = self.suffix

        #if (self.params_suffix != ''):
        #    suffix = suffix + '_' + self.params_suffix

        if (run_id == -1):
            # Set Run Id using timestamp
            dt = datetime.datetime.now()
            run_id = '{0}-{1}_{2}:{3}'.format(dt.month, dt.day, dt.hour, dt.minute)


        # location for the trained model file
        saved_model_file = os.path.join(model_save_dir,
                                        '{0}_{1}_{2}.h5'.format(
                                            run_id, self.name, suffix
                                        ))

        # location for the intermediate model files
        filepath = os.path.join(model_save_dir,
                                "{0}_{1}_{2}-model-improvement-".format(
                                    run_id, self.name, suffix,
                                ) + "{epoch:02d}-{val_acc:0.2f}.h5")

        # noqa, call back to save model files after each epoch (file saved only when the accuracy of the current epoch is max.)
        callback_checkpoint = ModelCheckpoint(
            filepath, monitor='val_acc', verbose=False,
            save_best_only=True, mode='max'
        )

        # noqa, sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam', metrics=['accuracy'])

        self.model.fit(X, Y, validation_data=val_data,
                       epochs=num_epochs, batch_size=batch_size,
                       callbacks=[callback_checkpoint], verbose=True)

        self.model.save(saved_model_file)

    def train(self, loader_func, run_id, batch_size=128, num_epochs=100):
        """
        Training procedure for a model.
        Loads data from dataloader and does supervised training.
        Saves intermediate models and the final models
        """

        loader_dict = loader_func(validation=True)
        data = loader_dict['data']
        self.num_classes = loader_dict['num_classes']
        model_save_dir = loader_dict['model_save_dir']
        suffix = loader_dict['name']
        #self.num_classes = loader_dict['num_classes']
        # Data is [x_train, y_train, x_test, y_test, x_val, y_val]
        print('self num classes', self.num_classes)
        self.fit(data[0], data[1], val_data=(data[4], data[5]),
                num_epochs=num_epochs, batch_size=batch_size,
                model_save_dir=model_save_dir, run_id = run_id, suffix = suffix)

        #accuracy = self.model.evaluate(
        #    data[2], data[3], batch_size=batch_size, verbose=False)
        y_pred = self.model.predict(data[2])
        scores = self.get_scores(data[3], y_pred=y_pred)
        print("Scores\n", scores)
        return scores['accuracy']

    def load_model_and_evaluate(self, loader_func, run_id, epoch=None, evaluate=True):
        loader_dict = loader_func(validation=True)
        data = loader_dict['data']
        self.num_classes = loader_dict['num_classes']
        model_save_dir = loader_dict['model_save_dir']
        suffix = loader_dict['name']
        if epoch is None:
            # Load a full model
            name = '{0}_{1}_{2}.h5'.format(run_id, self.name, suffix)
            filename = os.path.join(model_save_dir, name)
        else:
            # Load an improvement model
            name = "{0}_{1}_{2}-model-improvement-{3:02d}*.h5".format(
                run_id, self.name, suffix, epoch)
            search = glob.glob(os.path.join(model_save_dir, name))
            if len(search) == 0:
                print("No file with patter", name, "found")
                exit(0)
            elif len(search) > 1:
                print("Multiple files with pattern", name, "found")
                print(search)
                print("picking first file", search[0])

            filename = search[0]

        print("Filename", filename)
        self.model = load_model(filename)
        if (evaluate == True):
            if self.classifier is not None:
                self.init_encoder()
                self.classifier.fit(self.encoder.predict(data[0]), data[1])
                scores = self.get_scores(data[3], x_true=self.encoder.predict(data[2]))
            else:
                scores = self.get_scores(data[3], x_true=data[2])
            print(scores)

    def predict(self, X):
        if self.classifier is not None:
            # print(X.shape)
            return self.classifier.predict(X)
        return self.model.predict(X)

    def get_params(self, deep=True):
        """get_params needs t be modified in child classes
        to return respective params"""
        return {"name": self.name}

    def set_params(self, **parameters):
        for param, val in parameters.items():
            setattr(self, param, val)

        keys = list(parameters.keys())
        keys.sort() # to maintain order of params printed
        if 'suffix' in keys: keys.remove('suffix')
        s = self.suffix
        for key in keys:
            s = s + '_' + str(parameters[key])

    def get_scores(self, y_true, y_pred=None, x_true=None):
        if (y_pred is None and x_true is not None):
            y_pred = self.predict(x_true)
        elif(y_pred is None and x_true is None):
            print('invalid input, provide either y_pred or x_true')
            return
        if (len(y_true.shape) > 1):
            y_true = np.argmax(y_true, axis=1)
        if (len(y_pred.shape) > 1):
            y_pred = np.argmax(y_pred, axis=1)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),  # noqa
            # 'precision_micro': precision_score(y_true, y_pred, average='micro'),  # noqa
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            # 'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            # 'roc_auc': 
        }

    def score(self, X, y_true):
        y_pred = self.predict(X)
        if (len(y_true.shape) > 1):
            y_true = np.argmax(y_true, axis=1)
        if (len(y_pred.shape) > 1):
            y_pred = np.argmax(y_pred, axis=1)
        return accuracy_score(y_true, y_pred)


class CNN_GRU_Model(BaseModel):
    def __init__(self,name='CNN_GRU', num_gru=30, poolAvg=False,
            pool1D=False, pool2D=True, has2D=True, num_classes=4):
        super().__init__('CNN_GRU', num_classes)
        self.num_gru = 30
        self.poolAvg = poolAvg
        self.pool1D = pool1D
        self.has2D = has2D
        self.pool2D = pool2D


    def model_init(self, channels, observations, num_classes):
        """
        Model is essentially two 1D convolutions
        followed by one 2d Convolution
        fed into a GRU unit
        """
        self.num_classes = num_classes
        model = Sequential()
        model.add(InputLayer((channels, observations, 1)))
        model.add(Conv2D(4, (1, 32), padding='same', use_bias=False))
        model.add(BatchNormalization())
        if (self.pool1D):
            model.add(AveragePooling2D((1, 3)) if self.poolAvg else MaxPooling2D((1,3)))
        if (self.has2D):
            model.add(DepthwiseConv2D((channels, 1),
                                  depth_multiplier=2,
                                  use_bias=False,
                                  depthwise_constraint=max_norm(1.)))
            model.add(BatchNormalization())
            model.add(Activation('elu'))
        # model.add(Conv2D(50, (5, 4), activation='relu', data_format = 'channels_first')) # noqa
        if (self.has2D and self.pool2D):
            model.add(AveragePooling2D((1,3)) if self.poolAvg else MaxPooling2D((1, 3)))
        # model.add(Conv2D(100, (50, 2), activation='relu'))
        # , input_shape=(channels, observations)))
        model.add(TimeDistributed(GRU(self.num_gru)))
        # model.add(TimeDistributed(GRU(50)))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.25))
        model.add(BatchNormalization())
        model.add(Dense(self.num_classes, activation='softmax'))
        # print(model.summary())
        self.model = model
        self.enc_layer_ind = 6 #hardcoded for now..
    
    def load_model_and_evaluate(self, loader_func, run_id, epoch=None, evaluate=True):
        super().load_model_and_evaluate(loader_func, run_id, epoch=epoch, evaluate=evaluate)
        self.enc_layer_ind = 6


    def get_params(self, deep=True):
        params = super().get_params(deep)
        params['poolAvg'] = self.poolAvg
        params['pool1D'] = self.pool1D
        params['pool2D'] = self.pool2D
        params['has2D'] = self.has2D
        params['num_gru'] = self.num_gru
        return params



class CNN_Only_Model(BaseModel):
    def __init__(self):
        super().__init__('CNN_Only')

    def model_init(self, channels, observations, num_classes):
        model = Sequential()
        model.add(BatchNormalization(
            input_shape=(channels, observations, 1)))
        model.add(Conv2D(32, (1, 4), activation='relu'))
        model.add(Conv2D(25, (channels, 1), activation='relu'))
        model.add(MaxPooling2D((1, 3)))
        model.add(Conv2D(50, (4, 25),
                         activation='relu', data_format='channels_first'
                         ))
        model.add(MaxPooling2D((1, 3)))
        model.add(Conv2D(100, (50, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.7))
        model.add(Dense(self.num_classes, activation='softmax'))
        # print(model.summary())
        self.model = model
        self.enc_layer_ind = 8

    def load_model_and_evaluate(self, loader_func, run_id, epoch=None, evaluate=True):
        super().load_model_and_evaluate(loader_func, run_id, epoch=epoch, evaluate=evaluate)
        self.enc_layer_ind = 8


class AutoEncoder_Model(BaseModel):
    def __init__(self, classifier):
        super().__init__('AE')
        self.classifier = classifier

    def init_encoder(self):
        self.encoder = Model(inputs=self.model.input,
                             outputs=self.model.get_layer('embedding_output').output)

    def train(self, loader_func, run_id, batch_size=128, num_epochs=100):
        """
        Training procedure for a model.
        Loads data from dataloader and does supervised training.
        Saves intermediate models and the final models
        """

        loader_dict = loader_func(validation=True)
        data = loader_dict['data']
        self.num_classes = loader_dict['num_classes']
        model_save_dir = loader_dict['model_save_dir']
        suffix = loader_dict['name']
        num_classes = loader_dict['num_classes']
        # Data is [x_train, y_train, x_test, y_test]
        channels = data[0].shape[1]
        observations = data[0].shape[2]
        self.model_init(channels, observations, num_classes)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # location for the trained model file
        saved_model_file = os.path.join(model_save_dir,
                                        '{0}_{1}_{2}.h5'.format(
                                            run_id, self.name, suffix
                                        ))

        # location for the intermediate model files
        filepath = os.path.join(model_save_dir,
                                "{0}_{1}_{2}-model-improvement-".format(
                                    run_id, self.name, suffix,
                                ) + "{epoch:02d}-{val_loss:0.2f}.h5")

        # noqa, call back to save model files after each epoch (file saved only when the loss of the current epoch is min.)
        callback_checkpoint = ModelCheckpoint(
            filepath, monitor='val_loss', verbose=False,
            save_best_only=True, mode='min'
        )

        # noqa, sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adadelta')

        self.model.fit(data[0], data[0],
                       validation_data=(data[4], data[4]),
                       epochs=num_epochs, batch_size=batch_size,
                       callbacks=[callback_checkpoint], verbose=True)

        self.model.save(saved_model_file)

        self.classifier.fit(self.encoder.predict(data[0]), data[1])
        y_pred = self.classifier.predict(self.encoder.predict(data[2]))
        scores = self.get_scores(data[3], y_pred=y_pred)
        print("Scores\n", scores)
        return scores['accuracy']

    def model_init(self, channels, observations, num_classes):
        """
        Model is essentially two 1D convolutions
        followed by one 2d Convolution
        for extracting embeddings
        """
        model = Sequential()
        model.add(BatchNormalization(input_shape=(channels, observations, 1)))
        model.add(Conv2D(32, (1, 4), activation='relu'))
        model.add(Conv2D(25, (channels, 1), activation='relu'))
        model.add(MaxPooling2D((1, 3)))
        model.add(Conv2D(50, (4, 25),
                         activation='relu', data_format='channels_first'
                         ))
        model.add(MaxPooling2D((1, 3)))
        model.add(Conv2D(100, (50, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        reconstruction_shape = model.layers[-1].output_shape
        #         print(reconstruction_shape)
        # <--- 128 dimensional representation
        model.add(Dense(128, activation='relu', name="embedding_output"))
        # This concludes with features extraction, begin with upsampling now

        model.add(Dense(reconstruction_shape[1], activation='relu'))
        model.add(Dense(channels * observations, activation='relu'))
        model.add(Reshape((channels, observations, 1)))
        self.model = model
        self.encoder = Model(inputs=model.input,
                             outputs=model.get_layer('embedding_output').output)


class EEGNet_model(BaseModel):
    def __init__(self):
        super().__init__('EEGNet')

    def model_init(self, channels, observations, num_classes):
        F1 = 4
        F2 = 8
        D = 2
        norm_rate = 0.25
        kernLength = 32
        dropoutRate = 0.25

        input_layer = Input(shape=(channels, observations, 1))

        # originally bias=False was also present
        block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False,
                        input_shape=(channels, observations, 1))(input_layer)
        block1 = BatchNormalization(axis=1)(block1)
        block1 = DepthwiseConv2D((channels, 1), depth_multiplier=D,
                                 use_bias=False, depthwise_constraint=max_norm(1.))(block1)    # noqa
        block1 = BatchNormalization(axis=1)(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = Dropout(dropoutRate)(block1)

        block2 = SeparableConv2D(
            F2, (1, 16), use_bias=False, padding='same')(block1)
        block2 = BatchNormalization(axis=1)(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = Dropout(dropoutRate)(block2)

        flatten = Flatten(name='flatten')(block2)
        dense = Dense(num_classes, name='dense',
                      kernel_constraint=max_norm(norm_rate))(flatten)
        softmax = Activation('softmax', name='softmax')(dense)

        self.model = Model(inputs=input_layer, outputs=softmax)
        self.enc_layer_ind = 13

    def load_model_and_evaluate(self, loader_func, run_id, epoch=None, evaluate=True):
        super().load_model_and_evaluate(loader_func, run_id, epoch=epoch, evaluate=evaluate)
        self.enc_layer_ind = 13
