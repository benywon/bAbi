# -*- coding: utf-8 -*-
from DataProcessor.dataPreprocess import dataPreprocess
from public_functions import *

__author__ = 'benywon'


class SNLI(dataPreprocess):
    TRAIN_TOTAL = []

    def __init__(self,
                 **kwargs):
        dataPreprocess.__init__(self, **kwargs)
        self.path = self.path_base + 'snli_1.0/'
        append_str = '_batch' if self.batch_training else ''
        self.data_pickle_path = self.path + 'SNLI' + append_str + '.pickle'
        if self.reload:
            self.__build_data_set__()
        else:
            self.load_data()
        self.calc_data_stat()
        self.dataset_name = 'SNLI'

    def __build_data_set__(self):
        print 'start loading data from original file'
        trainfilepath = self.path + 'snli_1.0_train.txt'
        testfilepath = self.path + 'snli_1.0_test.txt'
        devfilepath = self.path + 'snli_1.0_dev.txt'

        def get_one_set(filepath):
            print 'process:' + filepath
            target = []
            with open(filepath, 'rb') as f:
                q = []
                yes = []
                no = []
                for line in f:
                    txts = line.split('\t')
                    label_str = txts[0]
                    premise = self.get_sentence_id_list(txts[5], max_length=self.Max_length)
                    hypothesis = self.get_sentence_id_list(txts[6], max_length=self.Max_length)
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

    def sample_data(self, sample_weight=0.5):
        if len(self.TRAIN_TOTAL) == 0:
            self.TRAIN_TOTAL = self.TRAIN
        print 'sample snli data with weight:' + str(sample_weight)
        if self.batch_training:
            length = len(self.TRAIN_TOTAL)
            sample_length = int(np.ceil(length * sample_weight))
            self.train_number = sample_length
            arr = np.arange(length)
            np.random.shuffle(arr)
            t = arr[0:sample_length]
            self.TRAIN = [self.TRAIN_TOTAL[i] for i in t]
        else:
            length = len(self.TRAIN_TOTAL[0])
            sample_length = int(np.ceil(length * sample_weight))
            self.train_number = sample_length
            arr = np.arange(length)
            np.random.shuffle(arr)
            t = arr[0:sample_length]
            q1 = [self.TRAIN_TOTAL[0][i] for i in t]
            q2 = [self.TRAIN_TOTAL[1][i] for i in t]
            q3 = [self.TRAIN_TOTAL[2][i] for i in t]
            self.TRAIN = [q1, q2, q3]
