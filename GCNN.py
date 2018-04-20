from __future__ import division, print_function, absolute_import
from keras import backend as K, optimizers
from keras.engine.topology import Layer, Input
from keras.layers import Conv1D, ZeroPadding1D, Multiply
from keras.models import Model

import numpy as np

class GatedConv1D:
    """Keras-style class implementation of a Gated CNN layer.
    See https://arxiv.org/abs/1612.08083 for more information.
    Example:    
        inp = Input(shape=(None, 9), ...)
        outp = GatedConv1D(output_dim=5, kernel_size=3)(inp)

    # Arguments
        ouput_dim (int)
        kernel_size (int)
        kwargs_conv (dict) : dictionary of keyword args for Keras Conv1D layer
        kwargs_gate (dict) : dictionary of keyword args for gated Keras Conv1D layer
    """

    def __init__(self, output_dim, kernel_size, kwargs_conv={}, kwargs_gate={}):
        self.conv = Conv1D(output_dim, kernel_size, **kwargs_conv) 
        self.conv_gate = Conv1D(output_dim, kernel_size, activation="sigmoid", **kwargs_gate)
        self.pad_input = ZeroPadding1D(padding=(kernel_size-1, 0))
    
    def __call__(self, inputs):
        X = self.pad_input(inputs)
        A = self.conv(X)
        B = self.conv_gate(X)
        return Multiply()([A, B])


import unittest

class TestGConv(unittest.TestCase):
    def setUp(self):
        kernel = 2
        input_shape = (1, 4, 6)
        output_dim = 3
        test_data = np.zeros(input_shape)
        inp = Input(shape=(None, input_shape[2]), name="input1")
        x = GatedConv1D(output_dim, kernel_size=kernel)(inp) 
        
        self.sample_zero_model = Model(inputs=inp, outputs=x)
        self.test_data = test_data
        self.output_dim = output_dim

    def test_forward_pass(self):
        model = self.sample_zero_model

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd)
        forward = model.predict(self.test_data, batch_size=None, verbose=1, steps=1)
        self.assertFalse(forward.any())

        model.fit(self.test_data, np.full(forward.shape, 1), epochs=10)
        forward = model.predict(self.test_data, batch_size=None, verbose=0, steps=1)
        self.assertTrue(forward.all())


if __name__ == "__main__":
    unittest.main()

