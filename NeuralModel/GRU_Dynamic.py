# -*- coding: utf-8 -*- 
__author__ = 'benywon'
import theano
import theano.tensor as T
sigmoid = lambda x: 1 / (1 + T.exp(-x))



def GRU_dynamic(embedding_in,attention_resource):
    def one_step(self, x_t, h_tm1, W_iz, W_hz, b_z, W_ir, W_hr, b_r, W_ih, W_hh, W_ho, b_o, b_h):
        zt = sigmoid(theano.dot(x_t, W_iz) + theano.dot(h_tm1, W_hz) + b_z)
        rt = sigmoid(theano.dot(x_t, W_ir) + theano.dot(h_tm1, W_hr) + b_r)
        rtht_1 = rt * h_tm1
        ht_hat = T.tanh(theano.dot(x_t, W_ih) + theano.dot(rtht_1, W_hh) + b_h)
        h_t = (1 - zt) * h_tm1 + zt * ht_hat
        y_t = theano.dot(h_t, W_ho) + b_o
        y_t = sigmoid(y_t)
        if self.ignore_zero:
            return [h_t, y_t], theano.scan_module.until(T.eq(T.sum(abs(x_t)), 0))
        return [h_t, y_t]
    outputs_list, _ = theano.scan(fn=one_step,
                                      sequences=[embedding_in],
                                      outputs_info=outputs_info,
                                      non_sequences=non_sequence)