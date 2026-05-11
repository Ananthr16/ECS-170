'''
Concrete IO class for Stage 3 image datasets (pickle format from instructor).
'''

import os
import pickle

import numpy as np

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading stage 3 image data...')

        path = (self.dataset_source_folder_path or '') + (self.dataset_source_file_name or '')
        if not os.path.isfile(path):
            path = os.path.join(self.dataset_source_folder_path or '', self.dataset_source_file_name or '')

        with open(path, 'rb') as f:
            packed = pickle.load(f)

        train_list = packed['train']
        test_list = packed['test']

        tag = (self.dataset_source_file_name or '').upper()
        if tag == 'MNIST':
            return self._stack_mnist(train_list, test_list)
        if tag == 'ORL':
            return self._stack_orl(train_list, test_list)
        if tag == 'CIFAR':
            return self._stack_cifar(train_list, test_list)

        raise ValueError('dataset_source_file_name must be MNIST, ORL, or CIFAR (pickle basename).')

    @staticmethod
    def _to_nchw_gray(images):
        n = len(images)
        a0 = np.asarray(images[0])
        if a0.ndim == 3:
            h, w = a0.shape[0], a0.shape[1]
        else:
            h, w = a0.shape[0], a0.shape[1]
        x = np.empty((n, 1, h, w), dtype=np.float32)
        for i, im in enumerate(images):
            g = np.asarray(im, dtype=np.float32)
            if g.ndim == 3:
                g = g[:, :, 0]
            if g.max() > 1.5:
                g = g / 255.0
            x[i, 0] = g
        return x

    @staticmethod
    def _to_nchw_rgb(images):
        n = len(images)
        a0 = np.asarray(images[0], dtype=np.uint8)
        h, w = a0.shape[0], a0.shape[1]
        x = np.empty((n, 3, h, w), dtype=np.float32)
        for i, im in enumerate(images):
            arr = np.asarray(im, dtype=np.float32)
            x[i] = np.transpose(arr, (2, 0, 1)) / 255.0
        return x

    def _stack_mnist(self, train_list, test_list):
        xt = self._to_nchw_gray([d['image'] for d in train_list])
        xe = self._to_nchw_gray([d['image'] for d in test_list])
        yt = np.array([int(d['label']) for d in train_list], dtype=np.int64)
        ye = np.array([int(d['label']) for d in test_list], dtype=np.int64)
        return {'train_X': xt, 'train_y': yt, 'test_X': xe, 'test_y': ye}

    def _stack_orl(self, train_list, test_list):
        xt = self._to_nchw_gray([d['image'] for d in train_list])
        xe = self._to_nchw_gray([d['image'] for d in test_list])
        yt = np.array([int(d['label']) for d in train_list], dtype=np.int64) - 1
        ye = np.array([int(d['label']) for d in test_list], dtype=np.int64) - 1
        return {'train_X': xt, 'train_y': yt, 'test_X': xe, 'test_y': ye}

    def _stack_cifar(self, train_list, test_list):
        xt = self._to_nchw_rgb([d['image'] for d in train_list])
        xe = self._to_nchw_rgb([d['image'] for d in test_list])
        yt = np.array([int(d['label']) for d in train_list], dtype=np.int64)
        ye = np.array([int(d['label']) for d in test_list], dtype=np.int64)
        return {'train_X': xt, 'train_y': yt, 'test_X': xe, 'test_y': ye}
