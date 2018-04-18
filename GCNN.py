from __future__ import division, print_function, absolute_import
from keras.layers import Conv1d, ZeroPadding1D

class GConv(Layer):
    def __init__(self, output_dim, kernel_size, padding="prepend", **kwargs):
        self.output_dim = output_dim
        self.pad_input = ZeroPadding1d(padding=(kernel_size-1,))
        self.conv = Conv1D(output_dim, kernel_size) 
        self.conv_gate = Conv1(output_dim, kernel_size, activation="sigmoid")

        super(GConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv_weights = self.conv.get_weights()
        self.conv_gate_weights = self.conv_gate.get_weights()

    def call(self, x):
        x = self.pad_input(x)
        A = self.conv(x)
        B = self.conv_gate(x)
        return A.multiply(B)

    def compute_output_shape(self, input_shape):
        return self.conv.output_shape
