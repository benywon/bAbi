# -*- coding: utf-8 -*-
import sys

from IAGRU import IAGRU
from QASent import QASentdataPreprocess
from WikiQA import WikiQAdataPreprocess
import numpy as np

from OAGRU import OAGRU
from insuranceQA import insuranceQAPreprocess

__author__ = 'benywon'

insuranceQA = 'insuranceQA'
WikiQA = 'WikiQA'
QASent = 'QASent'
IAGru = 'IAGru'
OAGru = 'OAGru'


class AnswerSelection(IAGRU, OAGRU, insuranceQAPreprocess, WikiQAdataPreprocess, QASentdataPreprocess):
    def __init__(self,
                 MODEL='IAGRU',
                 DATASET='insuranceQA',
                 **kwargs):
        if DATASET == insuranceQA:
            insuranceQAPreprocess.__init__(self, **kwargs)
        elif DATASET == WikiQA:
            WikiQAdataPreprocess.__init__(self, **kwargs)
        else:
            QASentdataPreprocess.__init__(self, **kwargs)
        if MODEL == IAGru:
            IAGRU.__init__(self, **kwargs)
        else:
            OAGRU.__init__(self, **kwargs)

    def Train(self):
        print 'start training ' + self.dataset_name + ' IAGRU...'
        for epoch in xrange(self.epochs):
            print 'start epoch:' + str(epoch)
            for i in xrange(self.train_number):
                batch_length = ''
                if self.batch_training:
                    question = self.TRAIN[i][0]
                    batch_length = ' batch size:' + str(len(question))
                    answer_yes = self.TRAIN[i][1]
                    answer_no = self.TRAIN[i][2]
                else:
                    question = self.TRAIN[0][i]
                    answer_yes = self.TRAIN[1][i]
                    answer_no = self.TRAIN[2][i]
                loss = self.train_function(question, answer_yes, answer_no)
                b = (
                    "Process\t" + str(i) + " in total:" + str(self.train_number) + batch_length + ' loss: ' + str(loss))
                sys.stdout.write('\r' + b)
            MAP, MRR = self.Test()
            append_name = self.dataset_name + '_MAP_' + str(MAP) + '_MRR+' + str(MRR)
            self.save_model(append=append_name)

    def Test(self):
        print 'start testing...'
        final_result_MAP = []
        final_result_MRR = []
        for one_pack in self.TEST:
            batch_result = []
            for one in one_pack:
                out = self.test_function(one[0], one[1])
                batch_result.append([out, one[2]])
            batch_result.sort(key=lambda x: x[0], reverse=True)
            result = 0.
            right_index = 0.
            first_one_position = -1.0
            for i, value in enumerate(batch_result):
                if value[1] == 1:
                    if first_one_position == -1.0:
                        first_one_position = (i + 1)
                    right_index += 1
                    result += right_index / (i + 1)
            final_result_MAP.append(result / right_index)
            final_result_MRR.append(1.0 / first_one_position)
        MAP = np.mean(np.asarray(final_result_MAP))
        MRR = np.mean(np.asarray(final_result_MRR))
        print 'final-result-MAP:' + str(MAP)
        print 'final-result-MRR:' + str(MRR)
        return MAP, MRR


if __name__ == '__main__':
    c = AnswerSelection(optmizer='adadelta', batch_training=True, sampling=5, reload=True, Margin=0.15,
                        use_the_last_hidden_variable=False, use_clean=True, epochs=50, Max_length=50,
                        N_hidden=150)
    c.Train()
