# -*- coding: utf-8 -*-

from ModelBase import ModelBase
from RNN import *
from public_functions import *

__author__ = 'benywon'
import theano.tensor as T
import theano

sigmoids = lambda x: 1 / (1 + T.exp(-x))


class IAGRU(ModelBase):
    def __init__(self, RNN_MODE='GRU', max_ave_pooling='ave', use_the_last_hidden_variable=False, Margin=0.1,
                 **kwargs):
        ModelBase.__init__(self, **kwargs)
        self.RNN_MODE = RNN_MODE
        self.Model_name = 'IAGRU'
        self.Margin = Margin
        self.use_the_last_hidden_variable = use_the_last_hidden_variable
        self.max_ave_pooling = max_ave_pooling
        self.model_file_base_path = './model/IAGRU/'
        assert len(self.wordEmbedding) > 0, 'you have not initiate data!!!'
        self.build_model()
        self.print_model_info(model_name='IAGRU')

    @ModelBase.print_model_info
    def print_model_info(self, model_name='IAGRU'):
        """
        remember to add model name when call this function
        :param model_name:
        :return:
        """
        print 'use the last hidden variable as output:\t' + str(self.use_the_last_hidden_variable)
        print 'max or ave pooling?\t' + str(self.max_ave_pooling)
        print 'Embedding size:\t' + str(self.EmbeddingSize)
        print 'dictionary size:\t' + str(self.vocabularySize)
        print 'Margin:\t' + str(self.Margin)
        print 'negative sample size:\t' + str(self.sampling)
        print 'RNN mode:\t' + self.RNN_MODE

    def build_model_sample(self, output_softmax, only_for_test=False):
        print 'start building model IAGRU sample...'
        In_question = T.ivector('in_question')
        In_answer_right = T.ivector('in_answer_right')
        In_answer_wrong = T.ivector('in_answer_wrong')
        EmbeddingMatrix = theano.shared(np.asanyarray(self.wordEmbedding, dtype='float64'), name='WordEmbedding', )
        in_question_embedding = EmbeddingMatrix[In_question]
        in_answer_right_embedding = EmbeddingMatrix[In_answer_right]
        in_answer_wrong_embedding = EmbeddingMatrix[In_answer_wrong]
        # this is the shared function

        if self.RNN_MODE == 'GRU':
            forward = GRU(N_hidden=self.N_hidden, N_in=self.EmbeddingSize)
            backward = GRU(N_hidden=self.N_hidden, N_in=self.EmbeddingSize, backwards=True)
        elif self.RNN_MODE == 'LSTM':
            forward = LSTM(N_hidden=self.N_hidden, N_in=self.EmbeddingSize)
            backward = LSTM(N_hidden=self.N_hidden, N_in=self.EmbeddingSize, backwards=True)
        else:
            forward = RNN(N_hidden=self.N_hidden, N_in=self.EmbeddingSize)
            backward = RNN(N_hidden=self.N_hidden, N_in=self.EmbeddingSize, backwards=True)

        def get_gru_representation(In_embedding):
            forward.build(In_embedding)
            backward.build(In_embedding)
            lstm_forward = forward.get_hidden()
            lstm_backward = backward.get_hidden()
            if self.use_the_last_hidden_variable:
                return T.concatenate([lstm_forward, lstm_backward[::-1]], axis=1)
            else:
                return T.concatenate([lstm_forward, lstm_backward], axis=1)

        def trans_representationfromquestion(In_embedding, question):
            sigmoid = sigmoids(T.dot(T.dot(In_embedding, attention_projection), question))
            softmax = T.nnet.softmax_graph(T.dot(T.dot(In_embedding, attention_projection), question))
            transMatrix = In_embedding.T * sigmoid
            return get_gru_representation(transMatrix.T), softmax

        def get_output(In_matrix):
            if self.use_the_last_hidden_variable:
                Oq = In_matrix[-1]
            else:
                if self.max_ave_pooling == 'ave':
                    Oq = T.mean(In_matrix, axis=0)
                else:
                    Oq = T.max(In_matrix, axis=0)
            return Oq

        attention_projection = theano.shared(sample_weights(self.EmbeddingSize, 2 * self.N_hidden),
                                             name='attention_projection')
        question_lstm_matrix = get_gru_representation(in_question_embedding)

        question_representation = get_output(question_lstm_matrix)

        answer_yes_lstm_matrix, softmax = trans_representationfromquestion(in_answer_right_embedding,
                                                                           question_representation)
        answer_no_lstm_matrix, _ = trans_representationfromquestion(in_answer_wrong_embedding, question_representation)

        oa_yes = get_output(answer_yes_lstm_matrix)
        oa_no = get_output(answer_no_lstm_matrix)

        all_params = forward.get_parameter()
        all_params.extend(backward.get_parameter())
        all_params.append(attention_projection)
        if self.classification:
            if self.N_out > 2:
                Wout = theano.shared(sample_weights(4 * self.N_hidden, self.N_out), name='Wout')
                representation = T.concatenate([question_representation, oa_yes], axis=0)
                prediction = T.nnet.softmax_graph(T.dot(representation, Wout))
                loss = T.nnet.categorical_crossentropy(prediction, In_answer_wrong)
                prediction_label = T.argmax(prediction)
                all_params.append(Wout)
            else:  # we can also use cosine similarity and transfer it into distribution
                Wout = theano.shared(sample_weights(4 * self.N_hidden, self.N_out), name='Wout')
                representation = T.concatenate([question_representation, oa_yes], axis=0)
                prediction = T.nnet.softmax_graph(T.dot(representation, Wout))
                loss = T.nnet.categorical_crossentropy(prediction, In_answer_wrong)
                prediction_label = T.argmax(prediction)
                all_params.append(Wout)


        else:
            predict_yes = cosine(oa_yes, question_representation)
            predict_no = cosine(oa_no, question_representation)

            margin = predict_yes - predict_no
            loss = T.maximum(0, self.Margin - margin)

        if self.Train_embedding:
            all_params.append(EmbeddingMatrix)
        self.parameter = all_params

        loss = self.add_l1_l2_norm(loss=loss)
        updates = self.get_update(loss=loss)

        print 'start compile function...'

        train = theano.function([In_question, In_answer_right, In_answer_wrong],
                                outputs=loss,
                                updates=updates,
                                allow_input_downcast=True)
        if output_softmax:
            test = theano.function([In_question, In_answer_right], outputs=softmax, on_unused_input='ignore',
                                   allow_input_downcast=True)
        else:
            test = theano.function([In_question, In_answer_right], outputs=predict_yes, on_unused_input='ignore',
                                   allow_input_downcast=True)
        print 'build model done!'
        return train, test

    def build_model_batch(self):
        print 'start building model IAGRU batch...'
        In_quesiotion = T.imatrix('in_question')
        In_answer_right = T.imatrix('in_answer_right')
        In_answer_wrong = T.imatrix('in_answer_wrong')
        EmbeddingMatrix = theano.shared(np.asanyarray(self.wordEmbedding, dtype='float64'), name='WordEmbedding', )
        in_question_embeddings = EmbeddingMatrix[In_quesiotion]
        in_answer_right_embeddings = EmbeddingMatrix[In_answer_right]
        in_answer_wrong_embeddings = EmbeddingMatrix[In_answer_wrong]
        # this is the shared function
        if self.RNN_MODE == 'GRU':
            forward = GRU(N_hidden=self.N_hidden, batch_mode=True, N_in=self.EmbeddingSize)
            backward = GRU(N_hidden=self.N_hidden, batch_mode=True, N_in=self.EmbeddingSize, backwards=True)
        elif self.RNN_MODE == 'LSTM':
            forward = LSTM(N_hidden=self.N_hidden, batch_mode=True, N_in=self.EmbeddingSize)
            backward = LSTM(N_hidden=self.N_hidden, batch_mode=True, N_in=self.EmbeddingSize, backwards=True)
        else:
            forward = RNN(N_hidden=self.N_hidden, batch_mode=True, N_in=self.EmbeddingSize)
            backward = RNN(N_hidden=self.N_hidden, batch_mode=True, N_in=self.EmbeddingSize, backwards=True)

        def get_gru_representation(In_embedding):
            forward.build(In_embedding)
            backward.build(In_embedding)
            lstm_forward = forward.get_hidden()
            lstm_backward = backward.get_hidden()
            if self.use_the_last_hidden_variable:
                return T.concatenate([lstm_forward, lstm_backward[::-1]], axis=2)
            else:
                return T.concatenate([lstm_forward, lstm_backward], axis=2)

        attention_projection = theano.shared(sample_weights(self.EmbeddingSize, 2 * self.N_hidden),
                                             name='attention_projection')
        question_lstm_matrix = get_gru_representation(in_question_embeddings)

        def get_output(In_matrix):
            if self.use_the_last_hidden_variable:
                Oq = In_matrix[-1]
            else:
                if self.max_ave_pooling == 'ave':
                    Oq = T.mean(In_matrix, axis=0)
                else:
                    Oq = T.max(In_matrix, axis=0)
            return Oq

        def trans_representationfromquestion(In_embedding, question):
            sigmoid = sigmoids(T.batched_dot(T.dot(In_embedding, attention_projection), question))
            transMatrix = In_embedding.dimshuffle(2, 0, 1) * sigmoid
            return get_gru_representation(transMatrix.dimshuffle(1, 2, 0))

        question_representation = get_output(question_lstm_matrix)
        answer_yes_lstm_matrix = trans_representationfromquestion(in_answer_right_embeddings, question_representation)
        answer_no_lstm_matrix = trans_representationfromquestion(in_answer_wrong_embeddings, question_representation)

        oa_yes = get_output(answer_yes_lstm_matrix)
        oa_no = get_output(answer_no_lstm_matrix)

        all_params = forward.get_parameter()
        all_params.extend(backward.get_parameter())
        all_params.append(attention_projection)
        if self.classification:
            if self.N_out > 2:
                Wout = theano.shared(sample_weights(4 * self.N_hidden, self.N_out), name='Wout')
                representation = T.concatenate([question_representation, oa_yes], axis=1)
                prediction = T.nnet.softmax_graph(T.dot(representation, Wout))
                loss = T.nnet.categorical_crossentropy(prediction, In_answer_wrong)
                prediction_label = T.argmax(prediction, axis=1)
                all_params.append(Wout)
            else:  # we can also use cosine similarity and transfer it into distribution
                Wout = theano.shared(sample_weights(4 * self.N_hidden, self.N_out), name='Wout')
                representation = T.concatenate([question_representation, oa_yes], axis=1)
                prediction = T.nnet.softmax_graph(T.dot(representation, Wout))
                loss = T.nnet.categorical_crossentropy(prediction, In_answer_wrong)
                prediction_label = T.argmax(prediction, axis=1)
                all_params.append(Wout)
            loss = T.mean(loss)
        else:
            predict_yes, _ = theano.scan(cosine, sequences=[oa_yes, question_representation])
            predict_no, _ = theano.scan(cosine, sequences=[oa_no, question_representation])

            margin = predict_yes - predict_no
            loss = T.mean(T.maximum(0, self.Margin - margin))

        self.parameter = all_params

        loss = self.add_l1_l2_norm(loss=loss)
        updates = self.get_update(loss=loss)
        print 'start compile function...'
        train = theano.function([In_quesiotion, In_answer_right, In_answer_wrong],
                                outputs=loss,
                                updates=updates,
                                allow_input_downcast=True)

        test = theano.function([In_quesiotion, In_answer_right],
                               outputs=prediction_label[0] if self.classification else predict_yes[0],
                               on_unused_input='ignore',
                               allow_input_downcast=True)
        print 'build model done!'
        return train, test


if __name__ == '__main__':
    ia = IAGRU(batch_training=True)
