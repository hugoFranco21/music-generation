from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import tensorflow.compat.v1 as tf

OCTAVE = 12
FIFTH = 7
THIRD = 4
SIXTH = 9
SECOND = 2
FOURTH = 5
SEVENTH = 11
HALFSTEP = 1

C_MAJOR_TONIC = 60

# Special intervals that have unique rewards
REST_INTERVAL = -1
HOLD_INTERVAL = -1.5
REST_INTERVAL_AFTER_THIRD_OR_FIFTH = -2
HOLD_INTERVAL_AFTER_THIRD_OR_FIFTH = -2.5
IN_KEY_THIRD = -3
IN_KEY_FIFTH = -5


def linear_annealing(n, total, p_initial, p_final):
    """Linearly interpolates a probability between p_initial and p_final.
    Current probability is based on the current step, n. Used to linearly anneal
    the exploration probability of the RLTuner.
    Args:
        n: The current step.
        total: The total number of steps that will be taken (usually the length of
        the exploration period).
        p_initial: The initial probability.
        p_final: The final probability.
    Returns:
        The current probability (between p_initial and p_final).
    """
    if n >= total:
        return p_final
    else:
        return p_initial - (n * (p_initial - p_final)) / (total)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sample_softmax(softmax_vect):
    """Samples a note from an array of softmax probabilities.
    Tries to do this with numpy, which requires that the probabilities add to 1.0
    with extreme precision. If this fails, uses a manual implementation.
    Args:
        softmax_vect: An array of probabilities.
    Returns:
        The index of the note that was chosen/sampled.
    """
    try:
        sample = np.argmax(np.random.multinomial(1, pvals=softmax_vect))
        return sample
    except:  # pylint: disable=bare-except
        r = random.uniform(0, np.sum(softmax_vect))
        upto = 0
        for i in range(len(softmax_vect)):
            if upto + softmax_vect[i] >= r:
                return i
            upto += softmax_vect[i]
        tf.logging.warn("Error! sample softmax function shouldn't get here")
        print("Error! sample softmax function shouldn't get here")
        return len(softmax_vect) - 1

def make_onehot(int_list, one_hot_length):
    """Convert each int to a one-hot vector.
    A one-hot vector is 0 everywhere except at the index equal to the
    encoded value.
    For example: 5 as a one-hot vector is [0, 0, 0, 0, 0, 1, 0, 0, 0, ...]
    Args:
        int_list: A list of ints, each of which will get a one-hot encoding.
        one_hot_length: The length of the one-hot vector to be created.
    Returns:
        A list of one-hot encodings of the ints.
    """
    return [[1.0 if j == i else 0.0 for j in range(one_hot_length)]
            for i in int_list]


def get_inner_scope(scope_str):
    """Takes a tensorflow scope string and finds the inner scope.
    Inner scope is one layer more internal.
    Args:
        scope_str: Tensorflow variable scope string.
    Returns:
        Scope string with outer scope stripped off.
    """
    idx = scope_str.find('/')
    return scope_str[idx + 1:]


def trim_variable_postfixes(scope_str):
    """Trims any extra numbers added to a tensorflow scope string.
    Necessary to align variables in graph and checkpoint
    Args:
        scope_str: Tensorflow variable scope string.
    Returns:
        Scope string with extra numbers trimmed off.
    """
    idx = scope_str.find(':')
    return scope_str[:idx]


def get_variable_names(graph, scope):
    """Finds all the variable names in a graph that begin with a given scope.
    Args:
        graph: A tensorflow graph.
        scope: A string scope.
    Returns:
        List of variables.
    """
    with graph.as_default():
        return [v.name for v in tf.global_variables() if v.name.startswith(scope)]


def get_next_file_name(directory, prefix, extension):
    """Finds next available filename in directory by appending numbers to prefix.
    E.g. If prefix is 'myfile', extenstion is '.png', and 'directory' already
    contains 'myfile.png' and 'myfile1.png', this function will return
    'myfile2.png'.
    Args:
        directory: Path to the relevant directory.
        prefix: The filename prefix to use.
        extension: String extension of the file, eg. '.mid'.
    Returns:
        String name of the file.
    """
    name = directory + '/' + prefix + '.' + extension
    i = 0
    while os.path.isfile(name):
        i += 1
        name = directory + '/' + prefix + str(i) + '.' + extension
    return name


def make_rnn_cell(rnn_layer_sizes, state_is_tuple=False):
    """Makes a default LSTM cell for use in the NoteRNNLoader graph.
    This model is only to be used for loading the checkpoint from the research
    paper. In general, events_rnn_graph.make_rnn_cell should be used instead.
    Args:
        rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
            RNN.
        state_is_tuple: A boolean specifying whether to use tuple of hidden matrix
            and cell matrix as a state instead of a concatenated matrix.
    Returns:
        A tf.rnn.rnn_cell.MultiRNNCell based on the given hyperparameters.
    """
    cells = []
    for num_units in rnn_layer_sizes:
        cell = tf.nn.rnn_cell.LSTMCell(
            num_units, state_is_tuple=state_is_tuple)
        cells.append(cell)

    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)

    return cell


def log_sum_exp(xs):
    """Computes the log sum exp value of a tensor."""
    maxes = tf.reduce_max(xs, keep_dims=True)
    xs -= maxes
    return tf.squeeze(maxes, [-1]) + tf.log(tf.reduce_sum(tf.exp(xs), -1))

def default_dqn_hparams():
    """Generates the default hparams for RLTuner DQN model."""
    return {
            'random_action_probability': 0.1,
            'store_every_nth': 1,
            'train_every_nth': 5,
            'minibatch_size': 32,
            'discount_rate': 0.95,
            'max_experience': 100000,
            'target_network_update_rate': 0.01
        }
