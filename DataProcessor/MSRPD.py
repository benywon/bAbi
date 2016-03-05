# -*- coding: utf-8 -*-
from DataProcessor.dataPreprocess import dataPreprocess

__author__ = 'benywon'


class MSRPD(dataPreprocess):
    def __init__(self,
                 **kwargs):
        dataPreprocess.__init__(self, **kwargs)
        self.path = self.path_base + 'QAsent/'
        append_str = '_batch' if self.batch_training else ''
