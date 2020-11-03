import tflib as lib

import numpy as np
import tensorflow as tf

def Embedding(name, inputs, vocab_size, hidden_size, embed=None):
    """
    inputs_shape: (..., vocab_size)
    """
    with tf.name_scope(name):
        embed_values = np.random.uniform(size=(vocab_size, hidden_size))
        if not embed:
            embed = lib.param(name, embed_values)
        return tf.nn.embedding_lookup(embed, inputs)
