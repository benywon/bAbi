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
        testfilepath = self.path + 'msr_paraphrase_test.txt'

        def get_one_set(filepath, train=False):
            print 'process:' + filepath
            target = []
            with open(filepath, 'rb') as f:
                lines = f.readlines()[1:-1]
                q = []
                yes = []
                no = []
                for line in lines:
                    divs = line.split('\t')
                    label_index = int(divs[0])
                    label = pad_index2distribution(label_index, 2)
                    sent1 = clean_str(divs[3])
                    sent2 = clean_str(divs[4])
                    sent1_ids = self.get_sentence_id_list(sent1, max_length=self.Max_length)
                    sent2_ids = self.get_sentence_id_list(sent2, max_length=self.Max_length)
                    q.append(sent1_ids)
                    yes.append(sent2_ids)
                    no.append(label)
            if train:
                q2 = q + yes
                yes2 = yes + q
                no2 = no * 2
            else:
                q2 = q
                yes2 = yes
                no2 = no
            target.append(q2)
            target.append(yes2)
            target.append(no2)
            return target

        self.TRAIN = get_one_set(trainfilepath, train=True)
        self.TEST = get_one_set(testfilepath)

        transfun_default = lambda z: np.asmatrix(z, dtype='int32') if self.batch_training else np.asarray(z,
                                                                                                          dtype='int32')

        self.TEST = [map(transfun_default, x) for x in self.TEST]
        self.transfer_data(add_dev=False)

        print 'load data done'


if __name__ == '__main__':
    c = MSRPD(reload=True, batch_training=False)
