# -*- coding: utf-8 -*-
import numpy as np

from DataProcessor.QASent import QASentdataPreprocess
from DataProcessor.QASentForServer import QASentForServerdataPreprocess
from DataProcessor.WikiQA import WikiQAdataPreprocess
from DataProcessor.insuranceQA import insuranceQAPreprocess
from NeuralModel.IAGRU import IAGRU
from NeuralModel.OAGRU import OAGRU, OAGRU_small
from TaskBase import TaskBases
from public_functions import dump_file

__author__ = 'benywon'

insuranceQA = 'insuranceQA'
WikiQA = 'WikiQA'
QASent = 'QASent'

IAGru = 'IAGru'
OAGru = 'OAGru'
OAGru_SMALL = 'OAGru_small'


class AnswerSelection(TaskBases):
    def __init__(self, MODEL=IAGru, DATASET=WikiQA, **kwargs):
        TaskBases.__init__(self)
        if DATASET == insuranceQA:
            self.Data = insuranceQAPreprocess(**kwargs)
        elif DATASET == WikiQA:
            self.Data = WikiQAdataPreprocess(**kwargs)
        else:
            self.Data = QASentdataPreprocess(**kwargs)
        if MODEL == IAGru:
            self.Model = IAGRU(data=self.Data, **kwargs)
        elif MODEL == OAGru_SMALL:
            self.Model = OAGRU_small(data=self.Data, **kwargs)
        else:
            self.Model = OAGRU(data=self.Data, **kwargs)

    @TaskBases.Train
    def Train(self):
        MAP, MRR = self.Test()
        append_name = self.Data.dataset_name + '_MAP_' + str(MAP) + '_MRR_' + str(MRR)
        self.Model.save_model(append_name)

    def Test(self):
        print '\nstart testing...'
        final_result_MAP = []
        final_result_MRR = []
        for one_pack in self.Data.TEST:
            batch_result = []
            for one in one_pack:
                out = self.Model.test_function(one[0], one[1])
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

    def output_softmax(self):
        print 'start output softmax'
        self.Model.load_model(
            self.Model.model_file_base_path + '03-09-13:23:30WikiQA_MAP_0.623834547368_MRR_0.636164544189.pickle')
        length = len(self.Data.TRAIN[0])
        pool_list = list()
        tran = lambda x: '_'.join(map(str, x.tolist()))
        softmax_pool = []
        for i in xrange(length):
            question = self.Data.TRAIN[0][i]
            pool_list.append(tran(question))
            samples = []
            answer_yes = self.Data.TRAIN[1][i]
            answer_no = self.Data.TRAIN[2][i]
            if tran(answer_yes) not in pool_list:
                samples.append(answer_yes)
            if tran(answer_no) not in pool_list:
                samples.append(answer_no)
            for sample in samples:
                pool_list.append(tran(sample))
                softmax = self.Model.test_function(question, sample)
                softmax_pool.append(softmax)
        dump_file(obj=softmax_pool, filepath='softmax_result123.pickle')


if __name__ == '__main__':
    c = AnswerSelection(optmizer='adadelta', MODEL=OAGru, DATASET=WikiQA, batch_training=False, sampling=3,
                        reload=True,
                        output_softmax=False,
                        Margin=0.15,
                        use_the_last_hidden_variable=False, use_clean=True, epochs=50, Max_length=50,
                        N_hidden=180)
    c.output_softmax()
