# -*- coding: utf-8 -*- 
__author__ = 'benywon'

"""
An theano implementation of recurrent neural network,
it could be used as an inner layer for your network,
all the input and output are theano tensor variables

---------------------------example:--------------------------------------------------------------
backward = GRU(N_hidden=self.N_hidden, batch_mode=True, N_in=self.EmbeddingSize, backwards=True)
backward.build(In_embedding) # In_embedding is your embedding matrix(theano.tensor)
lstm_backward = backward.get_hidden() #the GRU_RNN hidden variable(theano.tensor)
...
# when calculate parameter for theano updates
all_params.extend(backward.get_parameter())
------------------------------------------------------------------------------------------------
"""

"""
we recommend to use hard_sigmoid as inner activation
tanh for hidden-hidden function, any other implementation is available in theano.tensor.nnet.
so we omit it for convenience

problem and further development please contact: research@bingning.wang

"""

import numpy as np

import theano
import theano.tensor as T

dtype = theano.config.floatX

sigmoid = lambda x: 1 / (1 + T.exp(-x))

rng = np.random.RandomState(1991)

theano.config.exception_verbosity = 'high'


class RNN:
    """
    base class for all recurrent model
    """

    def __init__(self,
                 N_out=2,
                 N_in=None,
                 batch_mode=False,
                 W_initiation='svd',
                 N_hidden=50,
                 only_return_final=False,
                 backwards=False,
                 ignore_zero=False,
                 learn_hidden_init=True,
                 contain_output=False):
        self.batch_mode = batch_mode
        self.ignore_zero = ignore_zero
        self.contain_output = contain_output
        self.backwards = backwards
        self.only_return_final = only_return_final
        self.N_in = N_in
        self.N_out = N_out
        self.N_hidden = N_hidden
        self.W_initiation = W_initiation

        # standard rnn parameter
        self.b_h = theano.shared(np.zeros(N_hidden, dtype=dtype))
        self.W_ih = theano.shared(self.sample_weights(N_in, N_hidden))
        self.W_hh = theano.shared(self.sample_weights(N_hidden, N_hidden))
        self.W_ho = theano.shared(self.sample_weights(N_hidden, N_out))
        self.b_o = theano.shared(np.zeros(N_out, dtype=dtype))
        self.h0 = theano.shared(np.zeros(N_hidden, dtype=dtype))
        self.params = [self.W_ih, self.W_hh, self.b_h]
        self.y_vals = None
        if learn_hidden_init:
            self.params.append(self.h0)
        self.h_vals = None
        if self.contain_output:
            self.params.extend([self.W_ho, self.b_o])

    def set_output(self, out_list):
        [self.h_vals, self.y_vals] = out_list

    def build(self, tensor_in, n_step=None):
        if self.batch_mode:
            tensor_in = tensor_in.dimshuffle(1, 0, 2)
        if self.backwards:
            tensor_in = tensor_in[::-1]

        step_fun = self.one_step if self.contain_output else self.one_step_no_output  # this should be re-implement in sub class

        outputs_info = self.get_initiation(tensor_in)
        non_sequence = self.get_sequences()

        outputs_list, _ = theano.scan(fn=step_fun,
                                      n_steps=n_step,
                                      sequences=tensor_in,
                                      outputs_info=outputs_info,
                                      non_sequences=non_sequence)
        self.set_output(out_list=outputs_list)

    def get_output_list(self):
        return [self.h_vals, self.y_vals]

    def get_initiation(self, tensor_in):
        if self.batch_mode:
            return [T.alloc(self.h0, tensor_in.shape[1], self.N_hidden), None]
        else:
            return [self.h0, None]

    def get_sequences(self):
        return [self.W_ih, self.W_hh, self.b_h, self.W_ho, self.b_o]

    def one_step(self, x_t, h_tm1, W_ih, W_hh, b_h, W_ho, b_o):
        h_t = T.tanh(theano.dot(x_t, W_ih) + theano.dot(h_tm1, W_hh) + b_h)
        y_t = theano.dot(h_t, W_ho) + b_o
        y_t = sigmoid(y_t)
        if self.ignore_zero:
            return [h_t, y_t], theano.scan_module.until(T.eq(T.sum(abs(x_t)), 0))
        return [h_t, y_t]

    def one_step_no_output(self, x_t, h_tm1, W_ih, W_hh, b_h, W_ho, b_o):
        """
        function that did not calculate the output data
        """
        h_t = T.tanh(theano.dot(x_t, W_ih) + theano.dot(h_tm1, W_hh) + b_h)
        if self.ignore_zero:
            return [h_t, h_t], theano.scan_module.until(T.eq(T.sum(abs(x_t)), 0))
        return [h_t, h_t]

    def sample_weights(self, sizeX, sizeY):
        """
        it has been proved that the max singular value of a matirx can not
        exceed 1 for the non exploding RNN issues
        :param sizeY: the initiation matrix size y
        :param sizeX:the initiation matrix size x
        :return: the svd matrix remove max value
        """
        if self.W_initiation == 'random':
            return rng.normal([sizeX, sizeY])
        else:
            values = np.ndarray([sizeX, sizeY], dtype=dtype)
            for dx in xrange(sizeX):
                vals = np.random.uniform(low=-1., high=1., size=(sizeY,))
                # vals_norm = np.sqrt((vals**2).sum())
                # vals = vals / vals_norm
                values[dx, :] = vals
            _, svs, _ = np.linalg.svd(values)
            # svs[0] is the largest singular value
            values = values / svs[0]
            return values

    def get_parameter(self):
        return self.params

    def get_hidden(self):
        if self.ignore_zero:
            self.h_vals = self.h_vals[0:-1]
        if self.only_return_final:
            return self.h_vals[-1]
        else:
            # return self.h_vals[::-1] if self.backwards else self.h_vals
            return self.h_vals

    def get_output(self):
        if self.ignore_zero:
            self.y_vals = self.y_vals[0:-1]
        if self.only_return_final:
            return self.y_vals[-1]
        else:
            return self.y_vals[::-1] if self.backwards else self.y_vals


class LSTM(RNN):
    """
    this is my implementation of lstm
    borrow heavily from this blog
    http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/LSTM.php
    '''
    i t = tanh(W xi x t + W hi h t−1 + b i )
    j t = sigmoid(W xj x t + W hj h t−1 + b j )
    f t = sigmoid(W xf x t + W hf h t−1 + b f )
    o t = tanh(W xo x t + W ho h t−1 + b o )
    c t = c t−1  f t + i t  j t
    h t = tanh(c t )  o t

     is gate operation

    """

    def __init__(self,
                 b_i_init=(-0.5, 0.5),
                 b_o_init=(-0.5, 0.5),
                 b_f_init=(0., 1.),
                 act=T.tanh,
                 **kwargs):
        # init parent attributes
        RNN.__init__(self, **kwargs)
        self.act = act
        self.W_xi = theano.shared(self.sample_weights(self.N_in, self.N_hidden))
        self.W_hi = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.W_ci = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.b_i = theano.shared(np.cast[dtype](np.random.uniform(b_i_init[0], b_i_init[1], size=self.N_hidden)))
        self.W_xf = theano.shared(self.sample_weights(self.N_in, self.N_hidden))
        self.W_hf = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.W_cf = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.b_f = theano.shared(np.cast[dtype](np.random.uniform(b_f_init[0], b_f_init[1], size=self.N_hidden)))
        self.W_xc = theano.shared(self.sample_weights(self.N_in, self.N_hidden))
        self.W_hc = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.b_c = theano.shared(np.zeros(self.N_hidden, dtype=dtype))
        self.W_xo = theano.shared(self.sample_weights(self.N_in, self.N_hidden))
        self.W_ho = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.W_co = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.b_o = theano.shared(np.cast[dtype](np.random.uniform(b_o_init[0], b_o_init[1], size=self.N_hidden)))
        self.W_hy = theano.shared(self.sample_weights(self.N_hidden, self.N_out))
        self.b_y = theano.shared(np.zeros(self.N_out, dtype=dtype))
        self.c0 = theano.shared(np.zeros(self.N_hidden, dtype=dtype))
        self.h0 = T.tanh(self.c0)
        self.c_vals = None

        self.params = [self.W_xi, self.W_hi, self.W_ci, self.b_i, self.W_xf, self.W_hf, self.W_cf, self.b_f, self.W_xc,
                       self.W_hc, self.b_c, self.W_xo, self.W_ho, self.W_co, self.b_o, self.c0]
        if self.contain_output:
            self.params.extend([self.W_hy, self.b_y])

    def set_output(self, out_list):
        [self.h_vals, self.c_vals, self.y_vals] = out_list

    def get_initiation(self, tensor_in):
        if self.batch_mode:
            return [T.alloc(self.h0, tensor_in.shape[1], self.N_hidden),
                    T.alloc(self.c0, tensor_in.shape[1], self.N_hidden), None]
        else:
            return [self.h0, self.c0, None]

    def get_sequences(self):
        return [self.W_xi, self.W_hi, self.W_ci,
                self.b_i,
                self.W_xf,
                self.W_hf, self.W_cf, self.b_f,
                self.W_xc,
                self.W_hc,
                self.b_c, self.W_xo, self.W_ho, self.W_co, self.b_o, self.W_hy, self.b_y]

    def get_cell(self):
        if not self.only_return_final:
            return self.c_vals
        else:
            return self.c_vals[-1]

    def one_step(self, x_t, h_tm1, c_tm1, W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c, W_xy,
                 W_ho, W_cy, b_o, W_hy, b_y):
        """
        this is the inner step for calc lstm

        remember that we use sigma function to make sure the output normalized to 0-1

        :return: the hidden and c_t y_t state
        """
        i_t = sigmoid(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi) + theano.dot(c_tm1, W_ci) + b_i)
        f_t = sigmoid(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf) + theano.dot(c_tm1, W_cf) + b_f)
        c_t = f_t * c_tm1 + i_t * self.act(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c)
        o_t = sigmoid(theano.dot(x_t, self.W_xo) + theano.dot(h_tm1, W_ho) + theano.dot(c_t, self.W_co) + b_o)
        h_t = o_t * self.act(c_t)
        y_t = sigmoid(theano.dot(h_t, W_hy) + b_y)
        if self.ignore_zero:
            return [h_t, c_t, y_t], theano.scan_module.until(T.eq(T.sum(abs(x_t)), 0))
        return [h_t, c_t, y_t]

    def one_step_no_output(self, x_t, h_tm1, c_tm1, W_xi, W_hi, W_ci, b_i, W_xf, W_hf, W_cf, b_f, W_xc, W_hc, b_c,
                           W_xy,
                           W_ho, W_cy, b_o, W_hy, b_y):

        i_t = sigmoid(theano.dot(x_t, W_xi) + theano.dot(h_tm1, W_hi) + theano.dot(c_tm1, W_ci) + b_i)
        f_t = sigmoid(theano.dot(x_t, W_xf) + theano.dot(h_tm1, W_hf) + theano.dot(c_tm1, W_cf) + b_f)
        c_t = f_t * c_tm1 + i_t * self.act(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c)
        o_t = sigmoid(theano.dot(x_t, self.W_xo) + theano.dot(h_tm1, W_ho) + theano.dot(c_t, self.W_co) + b_o)
        h_t = o_t * self.act(c_t)
        if self.ignore_zero:
            return [h_t, c_t, o_t], theano.scan_module.until(T.eq(T.sum(abs(x_t)), 0))
        return [h_t, c_t, o_t]


class GRU(RNN):
    """
    r t = sigm (W xr x t + W hr h t−1 + b r )
    z t = sigm(W xz x t + W hz h t−1 + b z )
    h' t = tanh(W xh x t + W hh (r t  h t−1 ) + b h )
    h t = z t  h t−1 + (1 − z t )  h 't
    """

    def __init__(self,
                 b_i_init=(-0.5, 0.5),
                 act=T.tanh,
                 **kwargs):
        # init parent attributes
        RNN.__init__(self, **kwargs)
        self.act = act
        self.W_iz = theano.shared(self.sample_weights(self.N_in, self.N_hidden))
        self.W_hz = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.W_ir = theano.shared(self.sample_weights(self.N_in, self.N_hidden))
        self.W_hr = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.b_z = theano.shared(np.cast[dtype](np.random.uniform(b_i_init[0], b_i_init[1], size=self.N_hidden)))
        self.b_r = theano.shared(np.cast[dtype](np.random.uniform(b_i_init[0], b_i_init[1], size=self.N_hidden)))
        self.params.extend([self.W_iz, self.W_hz, self.W_ir, self.W_hr, self.b_z, self.b_r])

    def get_sequences(self):
        return [self.W_iz, self.W_hz, self.b_z, self.W_ir, self.W_hr,
                self.b_r, self.W_ih, self.W_hh, self.W_ho, self.b_o,
                self.b_h]

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

    def one_step_no_output(self, x_t, h_tm1, W_iz, W_hz, b_z, W_ir, W_hr, b_r, W_ih, W_hh, W_ho, b_o, b_h):
        """
        function that did not calculate the output data
        """
        zt = sigmoid(theano.dot(x_t, W_iz) + theano.dot(h_tm1, W_hz) + b_z)
        rt = sigmoid(theano.dot(x_t, W_ir) + theano.dot(h_tm1, W_hr) + b_r)
        rtht_1 = rt * h_tm1
        ht_hat = T.tanh(theano.dot(x_t, W_ih) + theano.dot(rtht_1, W_hh) + b_h)
        h_t = (1 - zt) * h_tm1 + zt * ht_hat
        if self.ignore_zero:
            return [h_t, h_t], theano.scan_module.until(T.eq(T.sum(abs(x_t)), 0))
        return [h_t, h_t]


class Highway(RNN):
    """
    it need to mention that the mechanism of Highway network could be
    implemented in any neural work other than RNN such as MLP. The carry-gate
    could carry the input information directly to the final representation
    which is really similar to,...emh,...highway...

    '''
    C= sigm (W xc x t + W hc h t−1 + b c ) # carry gate in Highway network
    h't = tanh(W xh x t + W hh h t−1 + b h )
    h t = h'tC + x(1-C)
    """

    def __init__(self,
                 b_i_init=(0, 1),
                 **kwargs):
        # init parent attributes
        RNN.__init__(self, **kwargs)
        self.W_xc = theano.shared(self.sample_weights(self.N_in, self.N_hidden))
        self.W_hc = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.b_c = theano.shared(np.cast[dtype](np.random.uniform(b_i_init[0], b_i_init[1], size=self.N_hidden)))
        self.params.extend([self.W_xc, self.W_hc, self.b_c])

    def get_sequences(self):
        return [self.W_xc, self.W_hc, self.b_c, self.W_ih, self.W_hh, self.W_ho, self.b_o,
                self.b_h]

    def one_step(self, x_t, h_tm1, W_xc, W_hc, b_c, W_ih, W_hh, W_ho, b_o, b_h):
        C = sigmoid(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c)
        h_t_hat = T.tanh(theano.dot(x_t, W_ih) + theano.dot(h_tm1, W_hh) + b_h)
        h_t = (1 - C) * h_t_hat + C * x_t
        y_t = theano.dot(h_t, W_ho) + b_o
        y_t = sigmoid(y_t)
        if self.ignore_zero:
            return [h_t, y_t], theano.scan_module.until(T.eq(T.sum(abs(x_t)), 0))
        return [h_t, y_t]

    def one_step_no_output(self, x_t, h_tm1, W_xc, W_hc, b_c, W_ih, W_hh, W_ho, b_o, b_h):
        C = sigmoid(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_c)
        h_t_hat = T.tanh(theano.dot(x_t, W_ih) + theano.dot(h_tm1, W_hh) + b_h)
        h_t = (1 - C) * h_t_hat + C * x_t
        if self.ignore_zero:
            return [h_t, h_t], theano.scan_module.until(T.eq(T.sum(abs(x_t)), 0))
        return [h_t, h_t]


class GRU_Attention(RNN):
    """
    wct=Mhc*h t-1 +Mqc
    act=sigm(wct*xt)
    x't=actxt
    r t = sigm (W xr x' t + W hr h t−1 + b r )
    z t = sigm(W xz x' t + W hz h t−1 + b z )
    h' t = tanh(W xh x' t + W hh (r t  h t−1 ) + b h )
    h t = z t  h t−1 + (1 − z t )  h 't
    """

    def __init__(self,
                 b_i_init=(-0.5, 0.5),
                 act=T.tanh,
                 **kwargs):
        # init parent attributes
        RNN.__init__(self, **kwargs)
        self.act = act
        self.W_iz = theano.shared(self.sample_weights(self.N_in, self.N_hidden))
        self.W_hz = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.W_ir = theano.shared(self.sample_weights(self.N_in, self.N_hidden))
        self.W_hr = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.b_z = theano.shared(np.cast[dtype](np.random.uniform(b_i_init[0], b_i_init[1], size=self.N_hidden)))
        self.b_r = theano.shared(np.cast[dtype](np.random.uniform(b_i_init[0], b_i_init[1], size=self.N_hidden)))
        # attention matrix
        self.M_hc = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.M_qc = theano.shared(self.sample_weights(self.N_hidden, self.N_hidden))
        self.w_c = theano.shared(np.cast[dtype](np.random.uniform(b_i_init[0], b_i_init[1], size=self.N_hidden)))

        self.params.extend(
            [self.W_iz, self.W_hz, self.W_ir, self.W_hr, self.b_z, self.b_r, self.M_hc, self.M_qc,
             self.w_c])
        self.attention_in = None

    def add_attention(self, attention_in):
        self.attention_in = attention_in

    def get_sequences(self):
        assert self.attention_in is not None, 'attention resource is none!!!!!'
        return [self.W_iz, self.W_hz, self.b_z, self.W_ir, self.W_hr,
                self.b_r, self.W_ih, self.W_hh, self.W_ho, self.b_o,
                self.b_h, self.M_hc, self.M_qc, self.w_c, self.attention_in]

    def one_step(self, x_t, h_tm1, W_iz, W_hz, b_z, W_ir, W_hr, b_r, W_ih, W_hh, W_ho, b_o, b_h, M_hc, M_qc, w_c,
                 attention_in):
        wct = T.dot(h_tm1, M_hc) + T.dot(attention_in, M_qc)
        act = sigmoid(T.dot(wct.T, x_t))
        x_t_hat = act * x_t
        zt = sigmoid(theano.dot(x_t_hat, W_iz) + theano.dot(h_tm1, W_hz) + b_z)
        rt = sigmoid(theano.dot(x_t_hat, W_ir) + theano.dot(h_tm1, W_hr) + b_r)
        rtht_1 = rt * h_tm1
        ht_hat = T.tanh(theano.dot(x_t_hat, W_ih) + theano.dot(rtht_1, W_hh) + b_h)
        h_t = (1 - zt) * h_tm1 + zt * ht_hat
        y_t = theano.dot(h_t, W_ho) + b_o
        y_t = sigmoid(y_t)
        if self.ignore_zero:
            return [h_t, y_t], theano.scan_module.until(T.eq(T.sum(abs(x_t)), 0))
        return [h_t, y_t]

    def one_step_no_output(self, x_t, h_tm1, W_iz, W_hz, b_z, W_ir, W_hr, b_r, W_ih, W_hh, W_ho, b_o, b_h, M_hc, M_qc,
                           w_c,
                           attention_in):
        """
        function that did not calculate the output data
        """
        print 'sttt'
        wct = T.dot(h_tm1, M_hc) + T.dot(attention_in, M_qc)
        act = sigmoid(T.dot(wct.T, x_t))
        x_t_hat = act * x_t
        zt = sigmoid(theano.dot(x_t_hat, W_iz) + theano.dot(h_tm1, W_hz) + b_z)
        rt = sigmoid(theano.dot(x_t_hat, W_ir) + theano.dot(h_tm1, W_hr) + b_r)
        rtht_1 = rt * h_tm1
        ht_hat = T.tanh(theano.dot(x_t_hat, W_ih) + theano.dot(rtht_1, W_hh) + b_h)
        h_t = (1 - zt) * h_tm1 + zt * ht_hat
        if self.ignore_zero:
            return [h_t, h_t], theano.scan_module.until(T.eq(T.sum(abs(x_t)), 0))
        return [h_t, h_t]


if __name__ == '__main__':
    # a = GRU_Attention(N_in=50, batch_mode=False, N_hidden=50, only_return_final=False, backwards=True)
    # # ain = np.ones(300,).reshape((5,6, 10))
    # ain = rng.normal(size=(10, 50))
    # atten = rng.normal(size=(50))
    # a.add_attention(atten)
    # # in_vector = T.tensor3('inv')  # this should be replaced by your theano shared variable or T input
    # in_vector = T.fmatrix('inv')  # this should be replaced by your theano shared variable or T input
    # N_0 = in_vector.shape[0]
    # a.build(in_vector)
    # hid = a.get_hidden()
    # params = a.get_parameter()  # to updates parameter use this params
    # fun = theano.function([in_vector], outputs=[hid, in_vector], allow_input_downcast=True)
    # hidden_output = fun(ain)
    # print hidden_output
    # print hidden_output
    a = Highway(N_in=50, batch_mode=False, N_hidden=50, only_return_final=False, backwards=True)
    # ain = np.ones(300,).reshape((5,6, 10))
    ain = rng.normal(size=(10, 50))
    # in_vector = T.tensor3('inv')  # this should be replaced by your theano shared variable or T input
    in_vector = T.fmatrix('inv')  # this should be replaced by your theano shared variable or T input
    N_0 = in_vector.shape[0]
    a.build(in_vector)
    hid = a.get_hidden()
    params = a.get_parameter()  # to updates parameter use this params
    fun = theano.function([in_vector], outputs=[hid, in_vector], allow_input_downcast=True)
    hidden_output = fun(ain)
    print hidden_output
    print hidden_output
