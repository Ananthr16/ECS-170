'''
Concrete IO class for Stage 2 dataset
'''

from local_code.base_class.dataset import dataset
import pandas as pd


class Dataset_Loader(dataset):
    data = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading stage 2 data...')

        train = pd.read_csv('data/stage_2_data/train.csv', header=None)
        test = pd.read_csv('data/stage_2_data/test.csv', header=None)

        train_y = train.iloc[:, 0].values
        train_X = train.iloc[:, 1:].values / 255.0

        test_y = test.iloc[:, 0].values
        test_X = test.iloc[:, 1:].values / 255.0

        return {
            'train_X': train_X,
            'train_y': train_y,
            'test_X': test_X,
            'test_y': test_y
        }