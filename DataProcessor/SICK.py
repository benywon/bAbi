# -*- coding: utf-8 -*-
from DataProcessor.dataPreprocess import dataPreprocess

__author__ = 'benywon'


class SICK(dataPreprocess):
    def __init__(self,
                 **kwargs):
        dataPreprocess.__init__(self, **kwargs)
        self.path = self.path_base + 'Semeval2014/'
        append_str = '_batch' if self.batch_training else ''
        self.data_pickle_path = self.path + 'SICK' + append_str + '.pickle'
        if self.reload:
            self.__build_data_set__()
        else:
            self.load_data()
        self.calc_data_stat()
        self.dataset_name = 'SICK'
