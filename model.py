# Set seeds before any other module imports to enable reproducible results
from numpy.random import seed
seed(20191129)
from tensorflow import set_random_seed
set_random_seed(20191129)

import os
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras import layers, utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt

import logging
import logging.handlers
from logging.config import dictConfig


ORIGINAL_DATA_DIR = os.path.join('.', 'bert/models/uncased_L-12_H-768_A-12/bert_input_data')
BERT_FEATURE_DIR = os.path.join('.', 'bert/models/uncased_L-12_H-768_A-12/bert_output_data')

plt.style.use('ggplot')


def logger():
    '''
    Custom logging config for readability/execution time tracking
    '''
    dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
        })
    logger = logging.getLogger(__name__)
    default_formatter = logging.Formatter((
        '[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s():%(lineno)s] '
        '[PID:%(process)d TID:%(thread)d] %(message)s'), '%Y/%m/%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(default_formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


class feature_data:
    def __init__(self):
        self.factors, self.k_factors, self.encoders, self.vectors = {}, {}, {}, {}
        self.num_classes = 0
        self.logger = logger()

    def import_data(self):
        '''
        Import features and class labels
        '''
        for dataset in ['train', 'test', 'eval']:
            # Encode classes into various formats for use by sklearn and tensorflow
            self.logger.info(f'Importing {dataset} labels')
            ser = pd.read_csv(os.path.join(ORIGINAL_DATA_DIR, f'{dataset}_label.txt'), header=None)[0]
            self.encoders[dataset] = LabelEncoder()
            self.encoders[dataset].fit(ser)
            self.factors[dataset] = self.encoders[dataset].transform(ser)
            self.k_factors[dataset] = utils.to_categorical(
                self.factors[dataset], len(ser.unique()))
            self.logger.info(f'{dataset} labels imported ({len(self.factors[dataset])} observations)')

            # Import features extracted via BERT
            self.logger.info(f'Importing {dataset} BERT vectors')
            with open(os.path.join(BERT_FEATURE_DIR, f'{dataset}.jsonlines'), 'rt') as infile:
                bert_vectors = []
                for line in infile:
                    bert_data = json.loads(line)
                    for t in bert_data['features']:
                        # Only extract the [CLS] vector used for classification
                        if t['token'] == '[CLS]':
                            # We only use the representation at the final layer of the network
                            bert_vectors.append(t['layers'][0]['values'])
                            break
            self.vectors[dataset] = np.array(bert_vectors)
            self.logger.info(f'{dataset} BERT vectors imported ({len(self.vectors[dataset])} observations)')
        self.num_classes = len(np.unique(self.factors['train']))

    def nn(self):
        '''
        Run the data through a neural network of low complexity
        '''
        # Model definition
        model = Sequential()
        model.add(layers.Dense(128, input_shape=self.vectors['train'][0].shape))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(self.k_factors['train'].shape[1]))
        model.add(layers.Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.001, momentum=0.9),
                      metrics=['accuracy'])
        model.summary()

        # Fit model to training dataset with evaluation dataset as the holdout
        history = model.fit(self.vectors['train'], self.k_factors['train'],
                            epochs=100,
                            verbose=False,
                            validation_data=(self.vectors['eval'], self.k_factors['eval']),
                            batch_size=32)
        self.plot_history(history)
        plt.savefig('nn_metrics.png')
        plt.close()
        self.logger.info('Neural network metrics saved to \'nn_metrics.png\'')

        # Accuracy comparison of training and test dataset
        loss, accuracy = model.evaluate(self.vectors['train'], self.k_factors['train'], verbose=False)
        print('\nTraining Accuracy: {:.4f}'.format(accuracy))
        loss, accuracy = model.evaluate(self.vectors['test'], self.k_factors['test'], verbose=False)
        print('Testing Accuracy: {:.4f}'.format(accuracy))

        # Metric preparation
        mcm = np.zeros((self.num_classes, self.num_classes))
        pred = model.predict_classes(self.vectors['test'])
        for i in range(len(pred)):
            mcm[self.factors['test'][i]][pred[i]] += 1
        width = max([len(x) for x in self.encoders['test'].classes_])
        fmt = '{0: >' + str(width) + '}'

        # Confusion matrix
        print('\nConfusion matrix (rows=actual, columns=predicted):\n')
        print(' ' * width, ' '.join([fmt.format(x) for x in self.encoders['test'].classes_]))
        for i in range(self.num_classes):
            row = mcm[i]
            row_str = []
            for j in range(self.num_classes):
                if i == j:
                    row_str.append(fmt.format('**' + str(int(row[j])) + '**'))
                else:
                    row_str.append(fmt.format(str(int(row[j]))))
            print(fmt.format(self.encoders['test'].classes_[i]), ' '.join(row_str))

        # Pairwise sensitivity by class
        print('\nPairwise sensitivity by class (rows=actual, columns=predicted):\n')
        print(' ' * width, ' '.join([fmt.format(x) for x in self.encoders['test'].classes_]))
        for i in range(self.num_classes):
            row = mcm[i] / mcm[:, i].sum()
            row_str = []
            for j in range(self.num_classes):
                if i == j:
                    row_str.append(fmt.format('**' + '{:.3f}'.format(row[j]) + '**'))
                else:
                    row_str.append(fmt.format('{:.3f}'.format(row[j])))
            print(fmt.format(self.encoders['test'].classes_[i]), ' '.join(row_str))

        # Pairwise recall by class
        print('\nPairwise recall by class (rows=actual, columns=predicted):\n')
        print(' ' * width, ' '.join([fmt.format(x) for x in self.encoders['test'].classes_]))
        for i in range(self.num_classes):
            row = mcm[i] / mcm[i].sum()
            row_str = []
            for j in range(self.num_classes):
                if i == j:
                    row_str.append(fmt.format('**' + '{:.3f}'.format(row[j]) + '**'))
                else:
                    row_str.append(fmt.format('{:.3f}'.format(row[j])))
            print(fmt.format(self.encoders['test'].classes_[i]), ' '.join(row_str))

    @staticmethod
    def plot_history(history):
        '''
        Tensorflow model analysis; visualizes various metrics over epochs
        '''
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()


if __name__ == '__main__':
    data = feature_data()

    # Simple menu for navigation
    running = 1
    while running:
        print('----------\nMenu:')
        for item in ['\t1. Import features',
                     '\t2. Apply a low-complexity neural network',
                     '\t3. Print required directory tree structure',
                     '\t0. Exit']:
            print(item)
        choice = input('\nSelection: ')
        print('\n----------')
        factors = 1
        for array in [data.factors, data.k_factors, data.encoders, data.vectors]:
            if not array:
                factors = 0
        if choice == '1':
            data.import_data()
        elif choice == '2':
            if factors == 0:
                print('Please import features before attempting modeling')
            else:
                data.nn()
        elif choice == '3':
            for item in ['.',
                         '  |-bert',
                         '     |-models',
                         '         |-uncased_L-12_H-768_A-12',
                         '             |-bert_input_data',
                         '                |-bert_prep.sh',
                         '                |-lang_id_eval.csv',
                         '                |-lang_id_test.csv',
                         '                |-lang_id_train.csv',
                         '                |-eval_data.txt',
                         '                |-eval_label.txt',
                         '                |-test_data.txt',
                         '                |-test_label.txt',
                         '                |-train_data.txt',
                         '                |-train_label.txt',
                         '             |-bert_output_data',
                         '                |-eval.jsonlines',
                         '                |-test.jsonlines',
                         '                |-train.jsonlines',
                         '  |-cs585_hw4_hile_andrew.py',
                         '  |-run_bert_fv.sh']:
                print(item)
        elif choice == '0':
            running = 0
        else:
            print('Invalid choice...')
