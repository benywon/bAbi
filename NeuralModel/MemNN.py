# -*- coding: utf-8 -*-
import sys
import time

import lasagne

from DataProcessor.bAbiDataPreprocess import bAbiDataPreprocess
from public_functions import *

__author__ = 'benywon'
import theano
import theano.tensor as T
import numpy as np


def softmax(inputs):
    return T.flatten(T.nnet.softmax(inputs))


class MemNN:
    def __init__(self, embedding_size=50,
                 learning_rate=0.01,
                 epoch=50,
                 stack=3,  # number of stacking layer if 0,no stack
                 hidden_vector_size=50):
        self.epoch = epoch
        self.Learning_rate = learning_rate
        self.embedding_size = embedding_size
        babidata = bAbiDataPreprocess()
        self.data = babidata.data
        self.word2id = babidata.word2id
        self.vocab_size = len(self.word2id)
        self.memnn_Parameter = None
        self.train_function, self.test_function = self.build_model() if stack == 0 else self.build_model_stacking(stack)


    def pad_answer(self, word_id):
        zeros = np.zeros(self.vocab_size)
        zeros[word_id] = 1
        return zeros

    def train(self):
        print 'start train'
        for j in xrange(self.epoch):
            for one_item_id in self.data:
                train_file = self.data[one_item_id]['train']
                for i, one_question in enumerate(train_file):
                    document = padding(one_question[1])[0]
                    question = np.asarray(one_question[2][0])
                    right_answer = self.pad_answer(one_question[2][1])
                    loss = self.train_function(document, question, right_answer)
                    b = (
                        'round:' + str(j) + ' Process\t' + str(i) + " in task:" + str(one_item_id) + ' loss: ' + str(
                            loss))
                    sys.stdout.write('\r' + b)
            this_err = self.test()
            self.save_parameter(append=this_err)

    def test(self):
        print 'start test....'
        total = 0
        total_right = 0.
        for one_item_id in self.data:
            test_file = self.data[one_item_id]['test']
            right = 0.
            for i, one_question in enumerate(test_file):
                total += 1
                document = padding(one_question[1])[0]
                question = np.asarray(one_question[2][0])
                right_answer = one_question[2][1]
                prediction = self.test_function(document, question)
                if right_answer == prediction:
                    right += 1
                    total_right += 1
            print 'task-' + str(one_item_id) + ' is:\t' + str(right / 1000)
        return total_right / total

    def build_model(self):
        print 'start build end-to-end memory network'
        document = T.imatrix('in_doc')
        question = T.ivector('in_question')
        right_answer = T.ivector('in_answer')

        def init_embedding(name):
            embedding = sample_weights(sizeX=self.vocab_size, sizeY=self.embedding_size)
            zeros = np.zeros(self.embedding_size, dtype=dtype)
            embedding[0] = zeros
            return theano.shared(embedding, name=name)

        embedding_B = init_embedding('embedding_B')  # for the output
        embedding_C = init_embedding('embedding_C')  # for the question
        embedding_A = init_embedding('embedding_A')  # for the softmax
        projection_matrix = theano.shared(sample_weights(sizeX=self.embedding_size, sizeY=self.vocab_size),
                                          name='projection')
        # build the M
        document_m_embedding = embedding_A[document]
        document_c_embedding = embedding_C[document]

        # then we calc the mean vector
        def get_mean_vec(docment_embedding, in_pre_embedding):
            def deal_oneline(line_embedding, inline):
                def sumscan(x, y):
                    return x, theano.scan_module.until(T.eq(y, 0))

                line, _ = theano.scan(sumscan, sequences=[line_embedding, inline])
                line = line[0:-1]
                return T.mean(line, axis=0)

            mean_vec_list, _ = theano.scan(deal_oneline, sequences=[docment_embedding, in_pre_embedding])
            return mean_vec_list

        document_m = get_mean_vec(document_m_embedding, document)
        document_c = get_mean_vec(document_c_embedding, document)

        question_representation = T.mean(embedding_B[question], axis=0)

        # then we calc the softmax vector

        softmax_layer = softmax(T.dot(document_m, question_representation.T))

        document_representation = T.dot(document_c.T, softmax_layer)

        # then we calc the sum representation

        representation = document_representation + question_representation

        predict = softmax(T.dot(projection_matrix.T, representation))

        prediction_id = T.argmax(predict)
        # calc loss

        loss = T.nnet.categorical_crossentropy(predict, right_answer)
        our_parameter = [embedding_A, embedding_B, embedding_C, projection_matrix]
        self.memnn_Parameter = our_parameter
        updates = lasagne.updates.sgd(loss, our_parameter, learning_rate=self.Learning_rate)

        print 'start compile train function...'

        train = theano.function([document, question, right_answer],
                                outputs=loss,
                                updates=updates,
                                allow_input_downcast=True,
                                on_unused_input='ignore')
        print 'start compile test function...'

        test = theano.function([document, question],
                               outputs=prediction_id,
                               allow_input_downcast=True,
                               on_unused_input='ignore')
        print 'build model done'
        return train, test

    def save_parameter(self, append=None):
        ISOTIMEFORMAT = '%Y-%m-%d-%X'
        now = time.strftime(ISOTIMEFORMAT, time.localtime())
        filename = str(now)
        if append is not None:
            filename += str(append)
        f = file('./model/bAbi/' + filename + '.pickle', 'wb')
        for parameter in self.memnn_Parameter:
            value = parameter.get_value()
            cPickle.dump(value, f, cPickle.HIGHEST_PROTOCOL)

    def load_parameter(self, modelname):
        print 'load model...' + modelname
        with open(modelname, 'rb') as f:
            for parameter in self.all_parameter:
                value = cPickle.load(f)
                parameter.set_value(value)

    def build_model_stacking(self, stack_number):
        print 'start build end-to-end memory network'
        document = T.imatrix('in_doc')
        question = T.ivector('in_question')
        right_answer = T.ivector('in_answer')

        def init_embedding():
            embedding = sample_weights(sizeX=self.vocab_size, sizeY=self.embedding_size)
            zeros = np.zeros(self.embedding_size, dtype=dtype)
            embedding[0] = zeros
            return theano.shared(embedding)

        embedding_B = init_embedding()  # for the output

        embedding_As = [init_embedding()]
        embedding_Cs = [init_embedding()]
        for i in range(1, stack_number + 1):
            embedding_Cs.append(init_embedding())
            embedding_As.append(embedding_Cs[i - 1])
        projection_matrix = theano.shared(sample_weights(sizeX=self.embedding_size, sizeY=self.vocab_size),
                                          name='projection')
        document_a_embeddings = []
        document_c_embeddings = []
        for i, embedding_A in enumerate(embedding_As):
            document_a_embeddings.append(embedding_A[document])
            document_c_embeddings.append(embedding_Cs[i][document])

        # then we calc the mean vector
        def get_mean_vec(docment_embedding, in_pre_embedding):
            def deal_oneline(line_embedding, inline):
                def sumscan(x, y):
                    return x, theano.scan_module.until(T.eq(y, 0))

                line, _ = theano.scan(sumscan, sequences=[line_embedding, inline])
                line = line[0:-1]
                return T.mean(line, axis=0)

            mean_vec_list, _ = theano.scan(deal_oneline, sequences=[docment_embedding, in_pre_embedding])
            return mean_vec_list

        u = T.mean(embedding_B[question], axis=0)
        for i in range(stack_number + 1):
            one_a_layer = document_a_embeddings[i]
            one_c_layer = document_c_embeddings[i]
            document_a = get_mean_vec(one_a_layer, document)
            document_c = get_mean_vec(one_c_layer, document)
            softmax_layer = softmax(T.dot(document_a, u.T))
            document_representation = T.dot(document_c.T, softmax_layer)
            representation = document_representation + u
            u = representation

        # then we calc the softmax vector



        predict = softmax(T.dot(projection_matrix.T, u))

        prediction_id = T.argmax(predict)
        # calc loss

        loss = T.nnet.categorical_crossentropy(predict, right_answer)
        our_parameter = [embedding_B, projection_matrix, embedding_As[0]]
        our_parameter.extend(embedding_Cs)
        self.memnn_Parameter = our_parameter
        updates = lasagne.updates.adadelta(loss, our_parameter)

        print 'start compile train function...'

        train = theano.function([document, question, right_answer],
                                outputs=loss,
                                updates=updates,
                                allow_input_downcast=True,
                                on_unused_input='ignore')
        print 'start compile test function...'

        test = theano.function([document, question],
                               outputs=prediction_id,
                               allow_input_downcast=True,
                               on_unused_input='ignore')
        print 'build model done'
        return train, test


if __name__ == '__main__':
    m = MemNN()
    m.train()
