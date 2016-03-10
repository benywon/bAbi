# -*- coding: utf-8 -*-
import random

from dataPreprocess import dataPreprocess
from public_functions import *

__author__ = 'benywon'


class insuranceQAPreprocess(dataPreprocess):
    def __init__(self,
                 neg_low=15,
                 neg_high=30,
                 Max_length=50,
                 **kwargs):
        dataPreprocess.__init__(self, **kwargs)
        self.Max_length = Max_length
        self.neg_high = neg_high
        self.neg_low = neg_low
        self.path = self.path_base + 'insuranceQA/'
        append_str = '_batch' if self.batch_training else ''
        self.data_pickle_path = self.path + 'insuranceQA' + append_str + '.pickle'
        if self.reload:
            self.__build_data_set__()
        else:
            self.load_data()
        self.calc_data_stat()
        self.dataset_name = 'insuranceQA'

    def __build_data_set__(self):
        print 'start loading data from original file'
        trainfilepath = self.path + 'question.train.token_idx.label'
        testfilepath1 = self.path + 'question.test1.label.token_idx.pool'
        testfilepath2 = self.path + 'question.test2.label.token_idx.pool'
        devfilepath = self.path + 'question.dev.label.token_idx.pool'
        answer_pool_path = self.path + 'answers.label.token_idx'
        vocabulary_path = self.path + 'vocabulary'

        def get_sentence_id(string):
            divs = string.split(' ')[0:self.Max_length]

            def get_word(word_str):
                return int(clean_str_remove(word_str.replace('idx_', '')))

            return map(get_word, divs)

        print 'start process answer pool....'
        answer_pool = {}
        with open(answer_pool_path, 'rb') as f:
            for line in f:
                divs = line.split('\t')
                id = divs[0]
                answer_sentence = get_sentence_id(divs[1])
                answer_pool[id] = answer_sentence
        pool_size = len(answer_pool)
        cc=np.mean([len(answer_pool[x]) for x in answer_pool])
        def get_test_or_dev_set(filepath):
            print 'process:' + filepath + '...'
            target = []
            with open(filepath, 'rb') as f:
                for line_question in f:
                    one_patch = []
                    divide = line_question.split('\t')
                    question = get_sentence_id(divide[1])
                    rights = divide[0].split(' ')
                    wrongs = divide[2].split(' ')
                    for right in rights:
                        yes = answer_pool[clean_str_remove(right.replace('idx_', ''))]
                        one_patch.append([self.transfun(question, 'int32'), self.transfun(yes, 'int32'), 1])
                    for wrong in wrongs:
                        no = answer_pool[clean_str_remove(wrong.replace('idx_', ''))]
                        one_patch.append([self.transfun(question, 'int32'), self.transfun(no, 'int32'), 0])
                    target.append(one_patch)
            return target

        self.TEST = get_test_or_dev_set(testfilepath1)
        self.DEV = get_test_or_dev_set(devfilepath)

        def get_neg_sample(id_in):
            id_in = clean_str_remove(id_in.replace('idx_', ''))
            sample_size = random.randint(self.neg_low, self.neg_high + 1)
            neg_pool = []
            for index in xrange(sample_size):
                sample_index = str(random.randint(1, pool_size))
                while id_in == sample_index:
                    sample_index = str(random.randint(1, pool_size))
                neg_pool.append(answer_pool[sample_index])
            return neg_pool

        print 'process:' + trainfilepath + '...'
        with open(trainfilepath, 'rb') as f:
            q = []
            yes = []
            no = []
            for line in f:
                dive = line.split('\t')
                question = get_sentence_id(dive[0])
                positives = dive[1].split(' ')
                for positive in positives:
                    pos = answer_pool[clean_str_remove(positive)]
                    negs = get_neg_sample(positive)
                    for neg in negs:
                        q.append(question)
                        yes.append(pos)
                        no.append(neg)
            self.TRAIN.append(q)
            self.TRAIN.append(yes)
            self.TRAIN.append(no)

        print 'train set length:' + str(len(self.TRAIN[0]))
        print 'start load:' + vocabulary_path + '...'
        with open(vocabulary_path, 'rb') as f:
            for word_str in f:
                divdes = word_str.split('\t')
                word_id = int(clean_str_remove(divdes[0].replace('idx_', '')))
                self.word2id[divdes[1]] = word_id

        self.transfer_data(add_dev=False)
        print 'load data done'

