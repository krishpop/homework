# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Krishnan Srinivasan <krishnan1994@gmail.com>
#
# Distributed under terms of the MIT license.
# ==============================================================================

"""
utils for training
"""

import numpy as np

class DataSet:
    """Base data set class
    """

    def __init__(self, shuffle=True, labeled=True, **data_dict):
        assert '_data' in data_dict
        if labeled:
            assert '_labels' in data_dict
            assert data_dict['_data'].shape[0] == data_dict['_labels'].shape[0]
        self._labeled = labeled
        self._shuffle = shuffle
        self.__dict__.update(data_dict)
        self._num_samples = self._data.shape[0]
        self._index_in_epoch = 0
        self._epochs_trained = 0
        self._batch_number = 0
        if self._shuffle:
            self._shuffle_data()

    def __len__(self):
        return len(self._data) + len(self._test_data)

    @property
    def epochs_trained(self):
        return self._epochs_trained

    @epochs_trained.setter
    def epochs_trained(self, new_epochs_trained):
        self._epochs_trained = new_epochs_trained

    @property
    def batch_number(self):
        return self._batch_number

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def labeled(self):
        return self._labeled

    @property
    def test_data(self):
        return self._test_data

    @property
    def test_labels(self):
        return self._test_labels

    @classmethod
    def load(cls, filename):
        data_dict = np.load(filename)
        shuffle = data_dict['_shuffle']
        labeled = data_dict['_labeled']
        return cls(shuffle=shuffle, labeled=labeled, **data_dict)

    def save(self, filename):
        data_dict = self.__dict__
        np.savez_compressed(filename, **data_dict)

    def _shuffle_data(self):
        shuffled_idx = np.arange(self._num_samples)
        np.random.shuffle(shuffled_idx)
        self._data = self._data[shuffled_idx]
        if self._labeled:
            self._labels = self._labels[shuffled_idx]

    def next_batch(self, batch_size):
        assert batch_size <= self._num_samples
        start = self._index_in_epoch
        if start + batch_size > self._num_samples:
            self._epochs_trained += 1
            self._batch_number = 0
            data_batch = self._data[start:]
            if self._labeled:
                labels_batch = self._labels[start:]
            remaining = batch_size - (self._num_samples - start)
            if self._shuffle:
                self._shuffle_data()
            start = 0
            data_batch = np.concatenate([data_batch, self._data[:remaining]],
                                        axis=0)
            if self._labeled:
                labels_batch = np.concatenate([labels_batch,
                                               self._labels[:remaining]],
                                              axis=0)
            self._index_in_epoch = remaining
        else:
            data_batch = self._data[start:start + batch_size]
            if self._labeled:
                labels_batch = self._labels[start:start + batch_size]
            self._index_in_epoch = start + batch_size
        self._batch_number += 1
        batch = (data_batch, labels_batch) if self._labeled else data_batch
        return batch


def make_dict_str(d={}, custom_keys=[], subset=[], kv_sep=': ', item_sep=', '):
    if not custom_keys:
        if subset:
            d = d.copy()
            subset = set(subset)
            for k in d.keys():
                if k not in subset:
                    del d[key]
        custom_keys = [(k,k) for k in d.keys()]
    custom_keys.sort(key=lambda x: x[0])

    item_list = []
    for c_key, key in custom_keys:
        item = d[key]
        if type(item) == float and item < 1e-4:
            item = '{:6.5E}'.format(item)
        elif type(item) == list:
            item = ','.join([str(x) for x in item])
        else:
            item = str(item)
        kv_str = kv_sep.join([c_key, item])
        item_list.append(kv_str)
    dict_str = item_sep.join(item_list)
    return dict_str
