# -*- coding: utf-8 -*- 
__author__ = 'benywon'

import theano
import theano.tensor as T
import numpy as np

dtype = theano.config.floatX

sigmoid = lambda x: 1 / (1 + T.exp(-x))

rng = np.random.RandomState(1991)

theano.config.exception_verbosity = 'high'