# -*- coding: utf-8 -*-
from ModelBase import ModelBase

from RNN import *
from public_functions import *

__author__ = 'benywon'


class OAGRU(ModelBase):
    def __init__(self, RNN_MODE='GRU', attention=True, max_ave_pooling='ave', use_the_last_hidden_variable=False,
                 Margin=0.1,
                 **kwargs):
        ModelBase.__init__(self, **kwargs)
        self.attention = attention
        self.Model_name = 'OAGRU'
        self.Margin = Margin
        self.RNN_MODE = RNN_MODE
        self.use_the_last_hidden_variable = use_the_last_hidden_variable
        self.max_ave_pooling = max_ave_pooling
        self.model_file_base_path = './model/OAGRU/'
        assert len(self.wordEmbedding) > 0, 'you have not initiate data!!!'
        self.build_model()
        self.print_model_info(model_name='OAGRU')

    @ModelBase.print_model_info
    def print_model_info(self, model_name='OAGRU'):
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

    def build_model_sample(self):
        """
        if you are in classfication mode
        In_quesiotion is the premise or the first sentence
        In_answer_right is the hypothesis or second sentence
        In_answer_wrong should be the true distribution
        :return:
        """
        print 'start building model OAGRU sample...'
        In_quesiotion = T.ivector('in_question')
        In_answer_right = T.ivector('in_answer_right')
        In_answer_wrong = T.ivector('in_answer_wrong')
        EmbeddingMatrix = theano.shared(np.asanyarray(self.wordEmbedding, dtype='float64'), name='WordEmbedding', )
        in_question_embedding = EmbeddingMatrix[In_quesiotion]
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

        def get_lstm_representation(In_embedding):
            forward.build(In_embedding)
            backward.build(In_embedding)
            lstm_forward = forward.get_hidden()
            lstm_bacward = backward.get_hidden()
            return T.concatenate([lstm_forward, lstm_bacward], axis=1)

        question_lstm_matrix = get_lstm_representation(in_question_embedding)
        answer_yes_lstm_matrix = get_lstm_representation(in_answer_right_embedding)
        answer_no_lstm_matrix = get_lstm_representation(in_answer_wrong_embedding)
        if self.max_ave_pooling == 'ave':
            Oq = T.mean(question_lstm_matrix, axis=0)
        else:
            Oq = T.max(question_lstm_matrix, axis=0)
        Wam = theano.shared(sample_weights(2 * self.N_hidden, 2 * self.N_hidden), name='Wam')
        Wms = theano.shared(rng.uniform(-0.3, 0.3, size=(2 * self.N_hidden)), name='Wms')
        Wqm = theano.shared(sample_weights(2 * self.N_hidden, 2 * self.N_hidden), name='Wqm')

        def get_final_result(answer_lstm_matrix):
            if not self.attention:
                Oa = T.mean(answer_lstm_matrix, axis=0)
            else:
                WqmOq = T.dot(Wqm, Oq)

                Saq_before_softmax = T.nnet.sigmoid(T.dot(answer_lstm_matrix, Wam) + WqmOq)

                Saq = T.nnet.softmax(T.dot(Saq_before_softmax, Wms))
                Oa = T.dot(T.flatten(Saq), answer_lstm_matrix)

            return Oa

        oa_yes = get_final_result(answer_yes_lstm_matrix)
        oa_no = get_final_result(answer_no_lstm_matrix)

        all_params = forward.get_parameter()
        all_params.extend(backward.get_parameter())

        if self.classfication:
            Wout = theano.shared(sample_weights(4 * self.N_hidden, self.N_out), name='Wout')
            representation = T.concatenate(Oq, oa_yes)
            prediction = T.dot(representation, Wout)
            loss = T.nnet.categorical_crossentropy(prediction, In_answer_wrong)
            all_params.append(Wout)
        else:
            predict_yes = cosine(oa_yes, Oq)
            predict_no = cosine(oa_no, Oq)

            margin = predict_yes - predict_no
            loss = T.maximum(0, self.Margin - margin)
        our_parameter = [Wam, Wms, Wqm]
        if self.attention:
            all_params.extend(our_parameter)

        if self.Train_embedding:
            all_params.append(EmbeddingMatrix)
        self.parameter = all_params

        loss = self.add_l1_l2_norm(loss=loss)
        updates = self.get_update(loss=loss)
        print 'compiling functions'

        train = theano.function([In_quesiotion, In_answer_right, In_answer_wrong],
                                outputs=loss,
                                updates=updates,
                                allow_input_downcast=True)
        test = theano.function([In_quesiotion, In_answer_right], outputs=predict_yes, on_unused_input='ignore')
        print 'build model done!'
        return train, test

    def build_model_batch(self):
        print 'start building model OAGRU batch...'
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

        Wam = theano.shared(sample_weights(2 * self.N_hidden, 2 * self.N_hidden), name='Wam')
        Wms = theano.shared(rng.uniform(-0.3, 0.3, size=(2 * self.N_hidden)), name='Wms')
        Wqm = theano.shared(sample_weights(2 * self.N_hidden, 2 * self.N_hidden), name='Wqm')

        def get_gru_representation(In_embedding):
            forward.build(In_embedding)
            backward.build(In_embedding)
            lstm_forward = forward.get_hidden()
            lstm_backward = backward.get_hidden()
            if self.use_the_last_hidden_variable:
                return T.concatenate([lstm_forward, lstm_backward[::-1]], axis=2)
            else:
                return T.concatenate([lstm_forward, lstm_backward], axis=2)

        question_lstm_matrix = get_gru_representation(in_question_embeddings)
        answer_yes_lstm_matrix = get_gru_representation(in_answer_right_embeddings)
        answer_no_lstm_matrix = get_gru_representation(in_answer_wrong_embeddings)

        def get_output(In_matrix):
            if self.use_the_last_hidden_variable:
                Oq = In_matrix[-1]
            else:
                if self.max_ave_pooling == 'ave':
                    Oq = T.mean(In_matrix, axis=0)
                else:
                    Oq = T.max(In_matrix, axis=0)
            return Oq

        def get_final_result(answer_lstm_matrix, question_representation):
            if not self.attention:
                Oa = T.mean(answer_lstm_matrix, axis=0)
            else:
                WqmOq = T.dot(question_representation, Wqm)
                Saq_before_softmax = T.nnet.sigmoid(T.dot(answer_lstm_matrix, Wam) + WqmOq)
                Saq = T.nnet.softmax(T.dot(Saq_before_softmax, Wms).T)
                Oa = T.batched_dot(Saq, answer_lstm_matrix.dimshuffle(1, 0, 2))
            return Oa

        question_representations = get_output(question_lstm_matrix)
        oa_yes = get_final_result(answer_yes_lstm_matrix, question_representations)
        oa_no = get_final_result(answer_no_lstm_matrix, question_representations)

        all_params = forward.get_parameter()
        all_params.extend(backward.get_parameter())

        if self.classfication:
            Wout = theano.shared(sample_weights(4 * self.N_hidden, self.N_out), name='Wout')
            representation = T.concatenate(question_representations, oa_yes)
            prediction = T.dot(representation, Wout)
            loss = T.nnet.categorical_crossentropy(prediction, In_answer_wrong)
            all_params.append(Wout)
        else:
            predict_yes, _ = theano.scan(cosine, sequences=[oa_yes, question_representations])
            predict_no, _ = theano.scan(cosine, sequences=[oa_no, question_representations])

            margin = predict_yes - predict_no
            loss = T.mean(T.maximum(0, self.Margin - margin))

        our_parameter = [Wam, Wms, Wqm]
        if self.attention:
            all_params.extend(our_parameter)

        if self.Train_embedding:
            all_params.append(EmbeddingMatrix)
        self.parameter = all_params

        loss = self.add_l1_l2_norm(loss=loss)
        updates = self.get_update(loss=loss)

        print 'start compile function...'
        train = theano.function([In_quesiotion, In_answer_right, In_answer_wrong],
                                outputs=loss,
                                updates=updates,
                                allow_input_downcast=True)

        test = theano.function([In_quesiotion, In_answer_right],
                               outputs=prediction[0] if self.classfication else predict_yes[0],
                               on_unused_input='ignore',
                               allow_input_downcast=True)
        print 'build model done!'
        return train, test


class OAGRU_small(ModelBase):
    """
    OAGRU model without much tensor product
    """

    def __init__(self, RNN_MODE='GRU', attention=True, max_ave_pooling='ave', use_the_last_hidden_variable=False,
                 Margin=0.1,
                 **kwargs):
        ModelBase.__init__(self, **kwargs)
        self.attention = attention
        self.Model_name = 'OAGRU_small'
        self.Margin = Margin
        self.RNN_MODE = RNN_MODE
        self.use_the_last_hidden_variable = use_the_last_hidden_variable
        self.max_ave_pooling = max_ave_pooling
        self.model_file_base_path = './model/OAGRU/samll'
        assert len(self.wordEmbedding) > 0, 'you have not initiate data!!!'
        self.build_model()
        self.print_model_info(model_name='OAGRU_small')

    @ModelBase.print_model_info
    def print_model_info(self, model_name='OAGRU'):
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

    def build_model_sample(self):
        """
        if you are in classfication mode
        In_quesiotion is the premise or the first sentence
        In_answer_right is the hypothesis or second sentence
        In_answer_wrong should be the true distribution
        :return:
        """
        print 'start building model OAGRU sample...'
        In_quesiotion = T.ivector('in_question')
        In_answer_right = T.ivector('in_answer_right')
        In_answer_wrong = T.ivector('in_answer_wrong')
        EmbeddingMatrix = theano.shared(np.asanyarray(self.wordEmbedding, dtype='float64'), name='WordEmbedding', )
        in_question_embedding = EmbeddingMatrix[In_quesiotion]
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

        def get_lstm_representation(In_embedding):
            forward.build(In_embedding)
            backward.build(In_embedding)
            lstm_forward = forward.get_hidden()
            lstm_bacward = backward.get_hidden()
            return T.concatenate([lstm_forward, lstm_bacward], axis=1)

        question_lstm_matrix = get_lstm_representation(in_question_embedding)
        answer_yes_lstm_matrix = get_lstm_representation(in_answer_right_embedding)
        answer_no_lstm_matrix = get_lstm_representation(in_answer_wrong_embedding)
        if self.max_ave_pooling == 'ave':
            Oq = T.mean(question_lstm_matrix, axis=0)
        else:
            Oq = T.max(question_lstm_matrix, axis=0)

        attention_projection = theano.shared(sample_weights(2 * self.N_hidden, 2 * self.N_hidden),
                                             name='attention_projection')

        def get_final_result(answer_lstm_matrix):
            if not self.attention:
                Oa = T.mean(answer_lstm_matrix, axis=0)
            else:
                weight_vector = T.nnet.softmax_graph(T.dot(T.dot(answer_lstm_matrix, attention_projection), Oq))

                Oa = T.dot(weight_vector, answer_lstm_matrix)

            return Oa

        oa_yes = get_final_result(answer_yes_lstm_matrix)
        oa_no = get_final_result(answer_no_lstm_matrix)

        all_params = forward.get_parameter()
        all_params.extend(backward.get_parameter())

        if self.classfication:
            Wout = theano.shared(sample_weights(4 * self.N_hidden, self.N_out), name='Wout')
            representation = T.concatenate(Oq, oa_yes)
            prediction = T.dot(representation, Wout)
            loss = T.nnet.categorical_crossentropy(prediction, In_answer_wrong)
            all_params.append(Wout)
        else:
            predict_yes = cosine(oa_yes, Oq)
            predict_no = cosine(oa_no, Oq)

            margin = predict_yes - predict_no
            loss = T.maximum(0, self.Margin - margin)
        our_parameter = [attention_projection]
        if self.attention:
            all_params.extend(our_parameter)

        if self.Train_embedding:
            all_params.append(EmbeddingMatrix)
        self.parameter = all_params

        loss = self.add_l1_l2_norm(loss=loss)
        updates = self.get_update(loss=loss)

        print 'compiling functions'

        train = theano.function([In_quesiotion, In_answer_right, In_answer_wrong],
                                outputs=loss,
                                updates=updates,
                                allow_input_downcast=True)
        test = theano.function([In_quesiotion, In_answer_right], outputs=predict_yes, on_unused_input='ignore')
        print 'build model done!'
        return train, test

    def build_model_batch(self):
        print 'start building model OAGRU batch...'
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

        Wam = theano.shared(sample_weights(2 * self.N_hidden, 2 * self.N_hidden), name='Wam')
        Wms = theano.shared(rng.uniform(-0.3, 0.3, size=(2 * self.N_hidden)), name='Wms')
        Wqm = theano.shared(sample_weights(2 * self.N_hidden, 2 * self.N_hidden), name='Wqm')

        def get_gru_representation(In_embedding):
            forward.build(In_embedding)
            backward.build(In_embedding)
            lstm_forward = forward.get_hidden()
            lstm_backward = backward.get_hidden()
            if self.use_the_last_hidden_variable:
                return T.concatenate([lstm_forward, lstm_backward[::-1]], axis=2)
            else:
                return T.concatenate([lstm_forward, lstm_backward], axis=2)

        question_lstm_matrix = get_gru_representation(in_question_embeddings)
        answer_yes_lstm_matrix = get_gru_representation(in_answer_right_embeddings)
        answer_no_lstm_matrix = get_gru_representation(in_answer_wrong_embeddings)

        def get_output(In_matrix):
            if self.use_the_last_hidden_variable:
                Oq = In_matrix[-1]
            else:
                if self.max_ave_pooling == 'ave':
                    Oq = T.mean(In_matrix, axis=0)
                else:
                    Oq = T.max(In_matrix, axis=0)
            return Oq

        def get_final_result(answer_lstm_matrix, question_representation):
            if not self.attention:
                Oa = T.mean(answer_lstm_matrix, axis=0)
            else:
                WqmOq = T.dot(question_representation, Wqm)
                Saq_before_softmax = T.nnet.sigmoid(T.dot(answer_lstm_matrix, Wam) + WqmOq)
                Saq = T.nnet.softmax(T.dot(Saq_before_softmax, Wms).T)
                Oa = T.batched_dot(Saq, answer_lstm_matrix.dimshuffle(1, 0, 2))
            return Oa

        question_representations = get_output(question_lstm_matrix)
        oa_yes = get_final_result(answer_yes_lstm_matrix, question_representations)
        oa_no = get_final_result(answer_no_lstm_matrix, question_representations)

        all_params = forward.get_parameter()
        all_params.extend(backward.get_parameter())

        if self.classfication:
            Wout = theano.shared(sample_weights(4 * self.N_hidden, self.N_out), name='Wout')
            representation = T.concatenate(question_representations, oa_yes)
            prediction = T.dot(representation, Wout)
            loss = T.nnet.categorical_crossentropy(prediction, In_answer_wrong)
            all_params.append(Wout)
        else:
            predict_yes, _ = theano.scan(cosine, sequences=[oa_yes, question_representations])
            predict_no, _ = theano.scan(cosine, sequences=[oa_no, question_representations])

            margin = predict_yes - predict_no
            loss = T.mean(T.maximum(0, self.Margin - margin))

        our_parameter = [Wam, Wms, Wqm]
        if self.attention:
            all_params.extend(our_parameter)

        if self.Train_embedding:
            all_params.append(EmbeddingMatrix)
        self.parameter = all_params

        loss = self.add_l1_l2_norm(loss=loss)
        updates = self.get_update(loss=loss)

        print 'start compile function...'
        train = theano.function([In_quesiotion, In_answer_right, In_answer_wrong],
                                outputs=loss,
                                updates=updates,
                                allow_input_downcast=True)

        test = theano.function([In_quesiotion, In_answer_right],
                               outputs=prediction[0] if self.classfication else predict_yes[0],
                               on_unused_input='ignore',
                               allow_input_downcast=True)
        print 'build model done!'
        return train, test
