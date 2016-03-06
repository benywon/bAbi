# -*- coding: utf-8 -*-
import sys

__author__ = 'benywon'


class TaskBases:
    def __init__(self):
        self.Data = None
        self.Model = None

    @classmethod
    def Train(cls, function):
        def wrapper(self, *args, **kwargs):
            print 'start training ' + self.Data.dataset_name + '  ' + self.Model.Model_name + '...'
            for epoch in xrange(self.Model.epochs):
                print 'start epoch:' + str(epoch)
                for i in xrange(self.Data.train_number):
                    batch_length = ''
                    if self.Data.batch_training:
                        question = self.Data.TRAIN[i][0]
                        batch_length = ' batch size:' + str(len(question))
                        answer_yes = self.Data.TRAIN[i][1]
                        answer_no = self.Data.TRAIN[i][2]
                    else:
                        question = self.Data.TRAIN[0][i]
                        answer_yes = self.Data.TRAIN[1][i]
                        answer_no = self.Data.TRAIN[2][i]
                    loss = self.Model.train_function(question, answer_yes, answer_no)
                    b = (
                        "Process\t" + str(i) + " in total:" + str(
                            self.Data.train_number) + batch_length + ' loss: ' + str(
                            loss))
                    sys.stdout.write('\r' + b)
                rst = function(self, *args, **kwargs)
            return rst

        return wrapper

    def IsIndexMatch(self, pre_arg, true_distribution):
        li = true_distribution.tolist()
        if li[pre_arg] == 1:
            return True
        return False

    def Test(self):
        raise Exception('You have not implement a test function')
