import tflib as lib

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops


def GRUCell(name, hidden_size, reuse=False):
    gru_cell = MyGRUCell(num_units=hidden_size, name=name, reuse=reuse)
    return gru_cell


_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class MyGRUCell(tf.nn.rnn_cell.GRUCell):
    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                            str(inputs_shape))
        _check_supported_dtypes(self.dtype)
        input_depth = inputs_shape[-1]
        # with tf.variable_scope(reuse=True):
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(self._bias_initializer
                        if self._bias_initializer is not None else
                        init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(self._bias_initializer
                        if self._bias_initializer is not None else
                        init_ops.zeros_initializer(dtype=self.dtype)))
        # print("----------------------------------")
        # print(self._gate_kernel.name)
        # print(self._gate_bias.name)
        # print(self._candidate_kernel.name)
        # print(self._candidate_bias.name)
        lib.add_param(self._gate_kernel.name, self._gate_kernel)
        lib.add_param(self._gate_bias.name, self._gate_bias)
        lib.add_param(self._candidate_kernel.name, self._candidate_kernel)
        lib.add_param(self._candidate_bias.name, self._candidate_bias)

        self.built = True


def _check_supported_dtypes(dtype):
    if dtype is None:
        return
    dtype = dtypes.as_dtype(dtype)
    if not (dtype.is_floating or dtype.is_complex):
        raise ValueError("RNN cell only supports floating point inputs, "
                        "but saw dtype: %s" % dtype)