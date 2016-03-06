# -*- coding: utf-8 -*-

from AnswerSelection import IAGru, OAGru_SMALL
from DataProcessor.MSRPD import MSRPD
from NeuralModel.IAGRU import IAGRU
from NeuralModel.OAGRU import OAGRU_small, OAGRU
from TaskBase import TaskBases

__author__ = 'benywon'

MSR = 'MSR'


class ParaphraseDetection(TaskBases):
    def __init__(self, MODEL=IAGru, DATASET=MSR, **kwargs):
        TaskBases.__init__(self)
        if DATASET == MSR:
            self.Data = MSRPD(**kwargs)

        if MODEL == IAGru:
            self.Model = IAGRU(data=self.Data, classfication=True, **kwargs)
        elif MODEL == OAGru_SMALL:
            self.Model = OAGRU_small(data=self.Data, classfication=True, **kwargs)
        else:
            self.Model = OAGRU(data=self.Data, classfication=True, **kwargs)

    def Test(self):
        print '\nstart testing...'
        length = len(self.Data.TEST[0])
        total = 0.
        right = 0.
        for i in xrange(length):
            question = self.Data.TEST[0][i]
            answer_yes = self.Data.TEST[1][i]
            prediction = self.Model.test_function(question, answer_yes)
            true = self.Data.TEST[2][i]
            total += 1
            if self.IsIndexMatch(prediction, true):
                right += 1
        precision = right / total
        print 'Precision is :\t' + str(precision)
        return precision

    @TaskBases.Train
    def Train(self):
        precision = self.Test()
        append_name = self.Data.dataset_name + '_Precision_' + str(precision)
        self.Model.save_model(append_name)


if __name__ == '__main__':
    c = ParaphraseDetection(optmizer='adadelta', MODEL=IAGru, DATASET=MSR, batch_training=False, sampling=3,
                            reload=False,
                            Margin=0.15,
                            N_out=2,
                            use_the_last_hidden_variable=False, epochs=50, Max_length=50,
                            N_hidden=50)
    c.Train()
