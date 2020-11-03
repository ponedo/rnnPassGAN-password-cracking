import tensorflow as tf
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.ops.gru_cell
import tflib.ops.embedding

def ResBlock(name, inputs, dim):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', dim, dim, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', dim, dim, 5, output)
    return inputs + (0.3*output)

def Generator(n_samples, seq_len, layer_dim, output_dim, prev_outputs=None):
    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, seq_len * layer_dim, output)
    output = tf.reshape(output, [-1, layer_dim, seq_len])
    output = ResBlock('Generator.1', output, layer_dim)
    output = ResBlock('Generator.2', output, layer_dim)
    output = ResBlock('Generator.3', output, layer_dim)
    output = ResBlock('Generator.4', output, layer_dim)
    output = ResBlock('Generator.5', output, layer_dim)
    output = lib.ops.conv1d.Conv1D('Generator.Output', layer_dim, output_dim, 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output, output_dim)
    return output

def Discriminator(inputs, seq_len, layer_dim, input_dim):
    output = tf.transpose(inputs, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', input_dim, layer_dim, 1, output)
    output = ResBlock('Discriminator.1', output, layer_dim)
    output = ResBlock('Discriminator.2', output, layer_dim)
    output = ResBlock('Discriminator.3', output, layer_dim)
    output = ResBlock('Discriminator.4', output, layer_dim)
    output = ResBlock('Discriminator.5', output, layer_dim)
    output = tf.reshape(output, [-1, seq_len * layer_dim])
    output = lib.ops.linear.Linear('Discriminator.Output', seq_len * layer_dim, 1, output)
    return output


def Generator_RNN1(n_samples, seq_len, rnn_layer, hidden_size, vocab_size, reuse=None):
    """
    noise_shape: (batch_size, 128)
    output_shape: (batch_size, seq_len, vocab_size)
    """
    cell = tf.nn.rnn_cell.MultiRNNCell([
        lib.ops.gru_cell.GRUCell('Generator.Rnn.' + str(i), hidden_size, reuse=reuse) 
        for i in range(rnn_layer)])

    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, seq_len * hidden_size, output)
    output = tf.reshape(output, (-1, seq_len, hidden_size))
    _outputs = tf.unstack(output, axis=1)

    state = cell.zero_state(n_samples, tf.float32)
    outputs = []
    for i in range(seq_len):
        output, state = cell(_outputs[i], state)
        outputs.append(output)
    output = tf.stack(outputs, axis=1)

    # outputs, _ = tf.nn.dynamic_rnn(multi_layer_cell, output, [float(seq_len) for _ in range(n_samples)], dtype=tf.float32)
    output = lib.ops.linear.Linear('Generator.Output', hidden_size, vocab_size, output) # reverse-embedding
    output = softmax(output, vocab_size)
    return output

def Discriminator_RNN1(inputs, seq_len, rnn_layer, hidden_size, vocab_size, reuse=None):
    """
    input_shape: (batch_size, seq_len, vocab_size)
    """
    multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([
        lib.ops.gru_cell.GRUCell('Discriminator.Rnn.' + str(i), hidden_size, reuse=reuse) 
        for i in range(rnn_layer)])

    # output = lib.ops.embedding.Embedding('Discriminator.Embedding', inputs, vocab_size, hidden_size)
    output = lib.ops.linear.Linear('Discriminator.Input', vocab_size, hidden_size, inputs)
    outputs, _ = tf.nn.dynamic_rnn(multi_layer_cell, inputs, [float(seq_len) for _ in range(inputs.shape[0])], dtype=tf.float32)
    output = tf.reshape(outputs, [-1, seq_len * hidden_size])
    output = lib.ops.linear.Linear('Discriminator.Output', seq_len * hidden_size, 1, output)
    return output


def Generator_RNN2(n_samples, seq_len, rnn_layer, hidden_size, vocab_size, reuse=None):
    """
    RNN decoder.
    noise_shape: (batch_size, 128)
    output_shape: (batch_size, seq_len, vocab_size)
    """
    cell = tf.nn.rnn_cell.MultiRNNCell([
        lib.ops.gru_cell.GRUCell('Generator.Rnn.' + str(i), hidden_size, reuse=reuse) 
        for i in range(rnn_layer)])
    
    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, hidden_size, output)
    state = cell.zero_state(n_samples, tf.float32)
    outputs = []
    for _ in range(seq_len):
        output, state = cell(output, state)
        outputs.append(output)
    output = tf.stack(outputs, axis=1)
    output = lib.ops.linear.Linear('Generator.Output', hidden_size, vocab_size, output)
    output = softmax(output, vocab_size)
    return output

def Discriminator_RNN2(inputs, seq_len, rnn_layer, hidden_size, vocab_size, reuse=None):
    """
    RNN encoder.
    input_shape: (batch_size, seq_len, vocab_size)
    """
    cell = tf.nn.rnn_cell.MultiRNNCell([
        lib.ops.gru_cell.GRUCell('Discriminator.Rnn.' + str(i), hidden_size, reuse=reuse) 
        for i in range(rnn_layer)])

    # output = lib.ops.embedding.Embedding('Discriminator.Input', inputs, vocab_size, hidden_size)
    output = lib.ops.linear.Linear('Discriminator.Input', vocab_size, hidden_size, inputs)
    outputs = tf.unstack(output, axis=1)
    state = cell.zero_state(inputs.shape[0], tf.float32)
    for i in range(seq_len):
        output, state = cell(outputs[i], state)
    output = lib.ops.linear.Linear('Discriminator.Output', hidden_size, 1, output)
    return output


def softmax(logits, num_classes):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random_normal(shape)
