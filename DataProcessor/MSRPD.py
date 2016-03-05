# -*- coding: utf-8 -*-
from DataProcessor.dataPreprocess import dataPreprocess
from public_functions import *

__author__ = 'benywon'


class MSRPD(dataPreprocess):
    def __init__(self,
                 **kwargs):
        dataPreprocess.__init__(self, **kwargs)
        self.path = self.path_base + 'MSA/'
        append_str = '_batch' if self.batch_training else ''
        self.data_pickle_path = self.path + 'MSA_paraphrase' + append_str + '.pickle'
        if self.reload:
            self.__build_data_set__()
        else:
            self.load_data()
        self.calc_data_stat()
        self.dataset_name = 'MSA'

    def __build_data_set__(self):
        print 'start loading data from original file'
        trainfilepath = self.path + 'msr_paraphrase_train.txt'
        testfilepath = self.path + 'WikiQA-test.tsv'
        devfilepath = self.path + 'WikiQA-dev.tsv'

        def get_one_set(filepath, train=True):
            print 'process:' + filepath
            with open(filepath, 'rb') as f:
                lines = f.readlines()[1:-1]
                q = []
                yes = []
                no = []
                for line in lines:
                    divs = line.split('\t')
                    label = int(divs[0])
                    sent1 = clean_str(divs[3])
                    sent2 = clean_str(divs[4])
                    sent1_ids = self.get_sentence_id_list(sent1, max_length=self.Max_length)
                    sent2_ids = self.get_sentence_id_list(sent2, max_length=self.Max_length)
                    q.append(sent1_ids)
                    yes.append(sent2_ids)

        self.TRAIN = get_one_set(trainfilepath, train=True)
        self.DEV = get_one_set(devfilepath, train=True)
        self.TEST = get_one_set(testfilepath, train=False)
        self.transfer_data()
        print 'load data done'
