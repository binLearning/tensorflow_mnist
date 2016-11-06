"""common operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from six.moves import xrange

import tensorflow as tf

import read_data

# Global constants describing the MNIST data set.
NUM_CLASSES = read_data.NUM_CLASSES
NUM_EXAMPLES_IN_TRAIN_SET = read_data.NUM_EXAMPLES_IN_TRAIN_SET

# for learning rate
BASE_LEARNING_RATE         = 0.01
NUM_EPOCHS_PER_DECAY       = 1
LEARNING_RATE_DECAY_FACTOR = 0.95


def _variable_on_cpu(name, shape, initializer):
  """Create a Variable stored on CPU memory.
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def variable_with_weight_decay(name, shape, stddev=1e-2,
                               coefficient=1e-3, useMultiGPU=False):
  """Create a Variable with weight decay.
  """
  if useMultiGPU:
    var = _variable_on_cpu(name, shape,
               tf.truncated_normal_initializer(stddev=stddev))
  else:
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)
  
  if coefficient is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var),
                          coefficient,
                          name='regularization_loss')
    tf.add_to_collection('losses', weight_decay)
  
  return var


def define_biases(name, shape, useMultiGPU=False):
  if useMultiGPU:
    biases = _variable_on_cpu(name, shape, tf.constant_initializer(0.0))
  else:
    biases = tf.Variable(tf.zeros(shape), name=name)
    
  return biases


def cross_entropy_loss(logits, labels, scheme=0):
  """Calculate cross entropy loss.
  """
  if scheme is 0:
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)

  if scheme is 1:
    labels_one_hot = tf.one_hot(labels, NUM_CLASSES, 1.0, 0.0)
    losses = tf.nn.softmax_cross_entropy_with_logits(logits, labels_one_hot)
  
  if scheme is 2:
    logits_log_softmax = -tf.nn.log_softmax(logits)
    labels_one_hot = tf.one_hot(labels, NUM_CLASSES, 1.0, 0.0)
    losses = tf.reduce_sum(labels * logits, reduction_indices=1)

  losses_mean = tf.reduce_mean(losses, name='cross_entropy_loss')
  tf.add_to_collection('losses', losses_mean)

  # total loss = cross entropy loss + regularization loss (L2 weight decay)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def optimizer(batch_size, global_step):
  """Optimizer.
  """
  
  num_batches_per_epoch = NUM_EXAMPLES_IN_TRAIN_SET // batch_size
  decay_steps = num_batches_per_epoch * NUM_EPOCHS_PER_DECAY
  
  # Decay the learning rate exponentially based on the number of steps.
  learning_rate = tf.train.exponential_decay(
      BASE_LEARNING_RATE,
      global_step,
      decay_steps,
      LEARNING_RATE_DECAY_FACTOR,
      staircase=True)
      
  # Use momentum for the optimizer
  optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
  
  return optimizer


def evaluate(logits, labels):
  """Calculate the accuracy of classifier.
  """
  #correct = tf.equal(tf.argmax(logits, 1), labels)
  #accuracy = tf.reduce_sum(tf.cast(correct, tf.float32))
  correct = tf.nn.in_top_k(logits, labels, 1)
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  return accuracy
