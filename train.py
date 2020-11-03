import argparse
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', '-i',
                        default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')

    parser.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')
    
    parser.add_argument('--model-type', '-m',
                        required=True,
                        dest='model_type',
                        help='Structure of discriminator and generator. Legal options: cnn, rnn, rnn1, rnn2')
    
    parser.add_argument('--checkpoint', '-ck',
                        dest='checkpoint',
                        help='Model checkpoint used to restore training. Expects a .ckpt file.')

    parser.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')

    parser.add_argument('--iters', '-n',
                        type=int,
                        default=200000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    parser.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length (default: 10)')
    
    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the ResNet generator and ResNet discriminator (default: 128)')
    
    parser.add_argument('--rnn-layer', '-r',
                        type=int,
                        default=2,
                        dest='rnn_layer',
                        help='The rnn layer number for the RNN generator and RNN discriminator (default: 1)')

    parser.add_argument('--hidden-size', '-hs',
                        type=int,
                        default=128,
                        dest='hidden_size',
                        help='The hidden layer dimensionality for the RNN generator and RNN discriminator (default: 128)')
    
    parser.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')
    
    parser.add_argument('--lambda', '-p',
                        type=int,
                        default=10,
                        dest='lamb',
                        help='The gradient penalty lambda hyperparameter (default: 10)')
    
    return parser.parse_args()

args = parse_args()
assert args.model_type in ["cnn", "rnn", "rnn1", "rnn2"], "Wrong model type. Legal types are: cnn, rnn."


import os, sys
sys.path.append(os.getcwd())

import time
import pickle
import numpy as np
import tensorflow as tf

import utils
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot
import models

DEBUG = False
myprint = print
if not DEBUG:
    print = lambda x: None

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', '-i',
                        default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')

    parser.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')
    
    parser.add_argument('--model-type', '-m',
                        required=True,
                        dest='model_type',
                        help='Structure of discriminator and generator. Legal options: cnn, rnn')
    
    parser.add_argument('--checkpoint', '-ck',
                        dest='checkpoint',
                        help='Model checkpoint used to restore training. Expects a .ckpt file.')

    parser.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')

    parser.add_argument('--iters', '-n',
                        type=int,
                        default=200000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    parser.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length (default: 10)')
    
    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the ResNet generator and ResNet discriminator (default: 128)')
    
    parser.add_argument('--rnn-layer', '-r',
                        type=int,
                        default=2,
                        dest='rnn_layer',
                        help='The rnn layer number for the RNN generator and RNN discriminator (default: 1)')

    parser.add_argument('--hidden-size', '-hs',
                        type=int,
                        default=128,
                        dest='hidden_size',
                        help='The hidden layer dimensionality for the RNN generator and RNN discriminator (default: 128)')
    
    parser.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')
    
    parser.add_argument('--lambda', '-p',
                        type=int,
                        default=10,
                        dest='lamb',
                        help='The gradient penalty lambda hyperparameter (default: 10)')
    
    return parser.parse_args()

args = parse_args()

assert args.model_type in ["cnn", "rnn", "rnn1", "rnn2"], "Wrong model type. Legal types are: cnn, rnn."

print("loading dataset lines, setting up vocabulary...")
ts = time.time() # recording timestamp
lines, charmap, inv_charmap = utils.load_dataset(
    path=args.training_data,
    max_length=args.seq_length
)

if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

if not os.path.isdir(os.path.join(args.output_dir, 'checkpoints')):
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'))

if not os.path.isdir(os.path.join(args.output_dir, 'samples')):
    os.makedirs(os.path.join(args.output_dir, 'samples'))

# pickle to avoid encoding errors with json
with open(os.path.join(args.output_dir, 'charmap.pickle'), 'wb') as f:
    pickle.dump(charmap, f)

with open(os.path.join(args.output_dir, 'inv_charmap.pickle'), 'wb') as f:
    pickle.dump(inv_charmap, f)

print("Dataset loading time: " + str(time.time() - ts))


# building calculating graph
print("Building calculating graph...")
ts = time.time()
real_inputs_discrete = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
if args.model_type == "cnn":
    fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))
elif args.model_type == "rnn1":
    fake_inputs = models.Generator_RNN1(args.batch_size, args.seq_length, args.rnn_layer, args.hidden_size, len(charmap))
elif args.model_type in ["rnn", "rnn2"]:
    fake_inputs = models.Generator_RNN2(args.batch_size, args.seq_length, args.rnn_layer, args.hidden_size, len(charmap))
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

if args.model_type == "cnn":
    disc_real = models.Discriminator(real_inputs, args.seq_length, args.layer_dim, len(charmap))
    disc_fake = models.Discriminator(fake_inputs, args.seq_length, args.layer_dim, len(charmap))
elif args.model_type == "rnn1":
    disc_real = models.Discriminator_RNN1(real_inputs, args.seq_length, args.rnn_layer, args.hidden_size, len(charmap))
    disc_fake = models.Discriminator_RNN1(fake_inputs, args.seq_length, args.rnn_layer, args.hidden_size, len(charmap), reuse=True)
elif args.model_type in ["rnn", "rnn2"]:
    disc_real = models.Discriminator_RNN2(real_inputs, args.seq_length, args.rnn_layer, args.hidden_size, len(charmap))
    disc_fake = models.Discriminator_RNN2(fake_inputs, args.seq_length, args.rnn_layer, args.hidden_size, len(charmap), reuse=True)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN lipschitz-penalty
alpha = tf.random_uniform(
    shape=[args.batch_size,1,1],
    minval=0.,
    maxval=1.
)

differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha*differences)
if args.model_type == "cnn":
    gradients = tf.gradients(models.Discriminator(interpolates, args.seq_length, args.layer_dim, len(charmap)), [interpolates])[0]
elif args.model_type == "rnn1":
    gradients = tf.gradients(models.Discriminator_RNN1(interpolates, args.seq_length, args.rnn_layer, args.hidden_size, len(charmap), reuse=True), [interpolates])[0]
elif args.model_type in ["rnn", "rnn2"]:
    gradients = tf.gradients(models.Discriminator_RNN2(interpolates, args.seq_length, args.rnn_layer, args.hidden_size, len(charmap), reuse=True), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += args.lamb * gradient_penalty

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

global_step = tf.Variable(tf.constant(0))

if args.model_type == "cnn":
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)
elif args.model_type == "rnn1":
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)
    # gen_train_op = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(gen_cost, var_list=gen_params)
    # disc_train_op = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(disc_cost, var_list=disc_params)
elif args.model_type in ["rnn", "rnn2"]:
    # gen_lr = tf.train.exponential_decay(1.0, global_step, decay_rate=0.99, decay_steps=100)
    # disc_lr = tf.train.exponential_decay(1.0, global_step, decay_rate=0.99, decay_steps=100)
    # gen_train_op = tf.train.AdamOptimizer(learning_rate=gen_lr, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    # disc_train_op = tf.train.AdamOptimizer(learning_rate=disc_lr, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

myprint("=======Trainable Params========")
for p in lib._params:
    myprint(p)

print("Graph building time: " + str(time.time() - ts))

# Dataset iterator
def traindata_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-args.batch_size+1, args.batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+args.batch_size]],
                dtype='int32'
            )

# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
print("Calculating initial JS divergence...")
ts = time.time() # recording timestamp
true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[10*args.batch_size:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines[:10*args.batch_size], tokenize=False) for i in range(4)]
for i in range(4):
    myprint("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [utils.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]
print("JS calculating time:" + str(time.time() - ts))


# Starting training session
saver = tf.train.Saver(max_to_keep=25)
with tf.Session() as session:

    # restore pretrained model and continue training
    restored_iteration = 0
    print("Restoring checkpoint...")
    if args.checkpoint:
        lib.plot.output_dir = args.output_dir
        saver.restore(session, args.checkpoint)
        restored_iteration = int(args.checkpoint.split("_")[-1][:-5])
        lib.plot.restore(restored_iteration)
    else:
        print("Initializing training variables...")
        ts = time.time()
        session.run(tf.global_variables_initializer())
        print("Variables initializing time: " + str(time.time() - ts))

    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    gen = traindata_gen()

    # Iterate starts
    for iteration in range(restored_iteration, args.iters):
        start_time = time.time()

        # Train GAN's generator
        if iteration > 0:
            print("-- Traning generator with batch...")
            ts = time.time()
            _gen_cost, _ = session.run(
                [gen_cost, gen_train_op], 
                feed_dict={global_step: iteration}
            )
            # myprint("gen cost: ", _gen_cost)
            print("-- Generator training time: " + str(time.time() - ts))

        # Train GAN's discriminator
        for i in range(args.critic_iters):
            print("-- Loading batch...")
            ts = time.time()
            _data = next(gen)
            print("-- Batch loading time: " + str(time.time() - ts))

            print("-- Traning discriminator with batch...")
            ts = time.time()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_inputs_discrete: _data, global_step: iteration}
            )
            print("-- Discriminator training time: " + str(time.time() - ts))

        lib.plot.output_dir = args.output_dir
        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc cost', _disc_cost)
        if iteration > 0:
            lib.plot.plot('train gen cost', _gen_cost)


        # output verbose info
        # use JS divergence to evaluate current language model generated by GAN
        if iteration % 100 == 0 and iteration > 0:
            print("-- Verbose JS calculating...")
            ts = time.time()
            # generate some generator samples first for calculating JS divergence 
            samples = []
            for i in range(int(64 / args.batch_size * 10)):
                samples.extend(generate_samples()) # num of generated samples: 640

            for i in range(4):
                lm = utils.NgramLanguageModel(i+1, samples, tokenize=False)
                lib.plot.plot('js{}'.format(i+1), lm.js_with(true_char_ngram_lms[i]))

            with open(os.path.join(args.output_dir, 'samples', 'samples_{}.txt').format(iteration), 'w') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")
            print("-- Verbose JS calculating: " + str(time.time() - ts))
        
        # save checkpoint
        if iteration % args.save_every == 0 and iteration > 0:
            # model_saver = tf.train.Saver()
            # model_saver.save(session, os.path.join(args.output_dir, 'checkpoints', 'checkpoint_{}.ckpt').format(iteration))
            saver.save(session, os.path.join(args.output_dir, 'checkpoints', 'checkpoint_{}.ckpt').format(iteration))

        if iteration % 100 == 0:
            lib.plot.flush()

        lib.plot.tick()
