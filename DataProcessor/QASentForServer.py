# -*- coding: utf-8 -*-


from dataPreprocess import dataPreprocess


__author__ = 'benywon'


class QASentForServerdataPreprocess(dataPreprocess):
    def __init__(self,
                 use_clean=False,
                 **kwargs):
        dataPreprocess.__init__(self, **kwargs)
        self.use_clean = use_clean
        self.path = self.path_base + 'QAsent/'
        append_str = '_batch' if self.batch_training else ''
        append_str += '_clean' if self.use_clean else ''
        self.data_pickle_path = self.path + 'QAsent' + append_str + '.pickle'
        if self.reload:
            self.build_data_set()
        else:
            self.load_data()
        self.calc_data_stat()
        self.dataset_name = 'QASent'