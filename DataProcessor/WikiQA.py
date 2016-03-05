# -*- coding: utf-8 -*-
from keras.preprocessing.sequence import pad_sequences

from dataPreprocess import dataPreprocess
from public_functions import *

__author__ = 'benywon'
import numpy as np

rng = np.random


class WikiQAdataPreprocess(dataPreprocess):
    def __init__(self,
                 sampling=3,
                 **kwargs):
        dataPreprocess.__init__(self, **kwargs)
        self.sampling = sampling
        self.using_YN_pair = None
        self.path = self.path_base + 'WikiQACorpus/'
        append_str = '_batch' if self.batch_training else ''
        self.data_pickle_path = self.path + 'wikiQA_sample' + str(sampling) + append_str + '.pickle'
        if self.reload:
            self.__build_data_set__()
        else:
            self.load_data()
        self.calc_data_stat()
        self.dataset_name = 'WikiQA'

    def __build_data_set__(self):
        print 'start loading data from original file'
        trainfilepath = self.path + 'WikiQA-train.tsv'
        testfilepath = self.path + 'WikiQA-test.tsv'
        devfilepath = self.path + 'WikiQA-dev.tsv'

        def get_one_set(filepath, sample=0, train=False):
            print 'process:' + filepath
            target = []
            questionPair = {}

            def deal_one_line(line, add_vacabulary=True):
                ts = line.split('\t')
                question_id = ts[0]
                question_origin = clean_str(ts[1])
                answer_origin = clean_str(ts[5])
                question = self.get_sentence_id_list(question_origin, add_vacabulary=add_vacabulary)
                answer = self.get_sentence_id_list(answer_origin, add_vacabulary=add_vacabulary)
                right = int(ts[6])
                if not question_id in questionPair:
                    questionPair[question_id] = list()
                questionPair[question_id].append([question, answer, right])

            with open(filepath, 'rb') as f:
                lines = f.readlines()
            [deal_one_line(x) for x in lines[1:]]
            at_least_one_right = [questionPair[x] for x in questionPair if sum([z[2] for z in questionPair[x]]) > 0]

            if not train:
                return [[[self.transfun(x[0], 'int32'), self.transfun(x[1], 'int32'), x[2]] for x in t] for t
                        in at_least_one_right]
            else:
                if sample > 0:
                    without_right_answer = [[z[1] for z in questionPair[x]] for x in questionPair if
                                            sum([z[2] for z in questionPair[x]]) == 0]
                q = []
                yes = []
                no = []
                for item in at_least_one_right:
                    question = item[0][0]
                    rights = [x[1] for x in item if x[2] == 1]
                    wrongs_origin = [x[1] for x in item if x[2] == 0]

                    for right in rights:
                        wrongs = wrongs_origin
                        if sample > 0:
                            random_number = rng.random_integers(0, len(at_least_one_right), size=sample)
                            [wrongs.extend(without_right_answer[i]) for i in random_number]
                        for wrong in wrongs:
                            q.append(question)
                            yes.append(right)
                            no.append(wrong)
                if self.padding_data:
                    target.append(pad_sequences(q, maxlen=self.Max_length, dtype='int32'))
                    target.append(pad_sequences(yes, maxlen=self.Max_length, dtype='int32'))
                    target.append(pad_sequences(no, maxlen=self.Max_length, dtype='int32'))
                else:
                    target.append([x[0:self.Max_length] for x in q])
                    target.append([x[0:self.Max_length] for x in yes])
                    target.append([x[0:self.Max_length] for x in no])
            return target

        self.DEV = get_one_set(filepath=devfilepath, sample=self.sampling, train=True)
        self.TRAIN = get_one_set(filepath=trainfilepath, sample=self.sampling, train=True)
        self.TEST = get_one_set(filepath=testfilepath)
        self.transfer_data()
        print 'load data done'


if __name__ == '__main__':
    WikiQAdataPreprocess(reload=True)
