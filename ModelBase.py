# -*- coding: utf-8 -*-
import time

import cPickle

import lasagne
import theano.tensor as T

__author__ = 'benywon'


class ModelBase:
    """
    this is the base to all model
    any model should be inherit from this class
    """

    def __init__(self, N_hidden=100, optmizer='sgd',
                 sampling=0,
                 learning_rate=0.01, l1=0.00001, l2=0.00001, epochs=50, **kwargs):
        self.sampling = sampling
        self.epochs = epochs
        self.l1 = l1
        self.l2 = l2
        self.optmizer = optmizer
        self.learning_rate = learning_rate
        self.N_hidden = N_hidden
        self.parameter = {}
        self.model_file_base_path = ''
        self.train_function = self.test_function = None

    @classmethod
    def print_model_info(cls, function):
        def wrapper(self, *args, **kwargs):
            print '--------------------Model information----------------------'
            model_name = kwargs['model_name']
            print 'MODEL NAME:\t' + str(model_name)
            print 'optmizer:\t' + str(self.optmizer)
            print 'hidden variable size:\t' + str(self.N_hidden)
            print 'learning rate:\t' + str(self.learning_rate)
            print 'l1:\t' + str(self.l1) + ' l2:\t' + str(self.l2)
            print 'epochs size:\t' + str(self.epochs)
            print 'batch training?:\t' + str(self.batch_training)
            if self.batch_training:
                print 'maximum batch size:\t'+self.max_batch_size
            rst = function(self, *args, **kwargs)
            print '------------------------------------------------------------'
            return rst

        return wrapper

    def build_model(self):
        """
        should be re-implement by child class
        :return: None
        """
        if self.batch_training:
            self.train_function, self.test_function = self.build_model_batch()
        else:
            self.train_function, self.test_function = self.build_model_sample()

    def build_model_sample(self):
        pass

    def build_model_batch(self):
        pass

    def save_model(self, append=None):
        isotimeformat = '%m-%d-%X'
        now = time.strftime(isotimeformat, time.localtime())
        filename = str(now)
        if append is not None:
            filename += append
        f = file(self.model_file_base_path + filename + '.pickle', 'wb')
        for parameter in self.parameter:
            value = parameter.get_value()
            cPickle.dump(value, f, cPickle.HIGHEST_PROTOCOL)

    def load_model(self, modelname):
        print 'load model...' + modelname
        with open(modelname, 'rb') as f:
            for parameter in self.parameter:
                value = cPickle.load(f)
                parameter.set_value(value)

    def get_update(self, loss):
        if self.optmizer == 'sgd':
            updates = lasagne.updates.sgd(loss, self.parameter, learning_rate=self.learning_rate)
        elif self.optmizer == 'adadelta':
            updates = lasagne.updates.adadelta(loss, self.parameter)
        elif self.optmizer == 'adagrad':
            updates = lasagne.updates.adagrad(loss, self.parameter)
        elif self.optmizer == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, self.parameter)
        else:
            updates = lasagne.updates.sgd(loss, self.parameter, learning_rate=self.learning_rate)
        return updates

    def add_l1_l2_norm(self, loss):
        for param in self.parameter:
            loss += self.l1 * T.sum(abs(param)) + self.l2 * T.sum(param ** 2)
        return loss
