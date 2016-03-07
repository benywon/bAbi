# -*- coding: utf-8 -*-
from DataProcessor.dataPreprocess import dataPreprocess
from public_functions import *

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

    def __build_data_set__(self):
        print 'start loading data from original file'
        trainfilepath = self.path + 'sick_train/SICK_train.txt'
        testfilepath = self.path + 'sick_test_annotated/SICK_test_annotated.txt'
        devfilepath = self.path + 'sick_trial/SICK_trial.txt'

        def get_one_set(filepath):
            print 'process:' + filepath
            target = []
            with open(filepath, 'rb') as f:
                q = []
                yes = []
                no = []
                for line in f:
                    txts = line.split('\t')
                    label_str = clean_str(txts[4])
                    premise = self.get_sentence_id_list(txts[1], max_length=self.Max_length)
                    hypothesis = self.get_sentence_id_list(txts[2], max_length=self.Max_length)
                    label_index = 0 if label_str == 'entailment' else 1 if label_str == 'neutral' else 2
                    label = pad_index2distribution(label_index, 3)
                    q.append(premise)
                    yes.append(hypothesis)
                    no.append(label)
            target.append(q)
            target.append(yes)
            target.append(no)
            return target

        self.TRAIN = get_one_set(trainfilepath)
        self.TEST = get_one_set(testfilepath)
        self.DEV = get_one_set(devfilepath)
        self.transferTest()
        self.transfer_data()
        print 'load data done'
