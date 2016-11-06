"""Various models."""

from __future__ import absolute_import
from __future__ import print_function

import math
from six.moves import xrange

import tensorflow as tf

import read_data
import common_ops

# Global constants describing the MNIST data set.
IMAGE_SIZE     = read_data.IMAGE_SIZE
IMAGE_CHANNELS = read_data.IMAGE_CHANNELS
NUM_PIXELS     = read_data.NUM_PIXELS
NUM_CLASSES    = read_data.NUM_CLASSES


def linear_classifier(images, useMultiGPU=False):
  """Linear classifier score function.
  f(x,W,b) = Wx + b
  """
  weights = common_ops.variable_with_weight_decay(
                'weight',
                [NUM_PIXELS, NUM_CLASSES],
                useMultiGPU=useMultiGPU)
  biases = common_ops.define_biases(
                'biases',
                [NUM_CLASSES],
                useMultiGPU=useMultiGPU)
  
  logits = tf.add(tf.matmul(images, weights), biases)
  
  return logits


def ann(images, useMultiGPU=False, hidden1_units=2048, hidden2_units=2048):
  """Traditional artificial neural network.
  """
  # hidden1
  with tf.variable_scope('hidden1'):
    weights = common_ops.variable_with_weight_decay(
                  'weight',
                  [NUM_PIXELS, hidden1_units],
                  stddev=1.0 / math.sqrt(float(NUM_PIXELS)),
                  useMultiGPU=useMultiGPU)
    biases = common_ops.define_biases(
                 'biases',
                 [hidden1_units],
                 useMultiGPU=useMultiGPU)
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # hidden2
  with tf.variable_scope('hidden2'):
    weights = common_ops.variable_with_weight_decay(
                  'weight',
                  [hidden1_units, hidden2_units],
                  stddev=1.0 / math.sqrt(float(hidden1_units)),
                  useMultiGPU=useMultiGPU)
    biases = common_ops.define_biases(
                 'biases',
                 [hidden2_units],
                 useMultiGPU=useMultiGPU)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # output
  with tf.variable_scope('output'):
    weights = common_ops.variable_with_weight_decay(
                  'weight',
                  [hidden2_units, NUM_CLASSES],
                  stddev=1.0 / math.sqrt(float(hidden2_units)),
                  useMultiGPU=useMultiGPU)
    biases = common_ops.define_biases(
                 'biases',
                 [NUM_CLASSES],
                 useMultiGPU=useMultiGPU)
    logits = tf.matmul(hidden2, weights) + biases
  
  return logits


def cnn(images, batch_size, useMultiGPU=False):
  """Convolutional neural network.
  """
  # conv1
  with tf.variable_scope('conv1'):
    kernel = common_ops.variable_with_weight_decay(
                 'weight',
                 [5,5,IMAGE_CHANNELS,32],
                 coefficient=0.0,
                 useMultiGPU=useMultiGPU)
    biases = common_ops.define_biases(
                 'biases',
                 [32],
                 useMultiGPU=useMultiGPU)
    conv1 = tf.nn.conv2d(images,
                         kernel,
                         strides=[1,1,1,1],
                         padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))
    pool1 = tf.nn.max_pool(relu1,
                           ksize=[1,2,2,1],
                           strides=[1,2,2,1],
                           padding='SAME')
  # conv2
  with tf.variable_scope('conv2'):
    kernel = common_ops.variable_with_weight_decay(
                 'weight',
                 [5,5,32,64],
                 coefficient=0.0,
                 useMultiGPU=useMultiGPU)
    biases = common_ops.define_biases(
                 'biases',
                 [64],
                 useMultiGPU=useMultiGPU)
    conv2 = tf.nn.conv2d(pool1,
                         kernel,
                         strides=[1,1,1,1],
                         padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases))
    pool2 = tf.nn.max_pool(relu2,
                           ksize=[1,2,2,1],
                           strides=[1,2,2,1],
                           padding='SAME')
  # fc (fully connected)
  with tf.variable_scope('fc'):
    # reshape the feature map cuboid into a 2D matrix to feed it to the fc layer.
    reshape = tf.reshape(pool2, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = common_ops.variable_with_weight_decay(
                  'weight',
                  [dim, 512],
                  stddev=1.0 / math.sqrt(float(dim)),
                  useMultiGPU=useMultiGPU)
    biases = common_ops.define_biases(
                 'biases',
                 [512],
                 useMultiGPU=useMultiGPU)
    fc = tf.nn.relu(tf.matmul(reshape, weights) + biases)
  # fc output
  with tf.variable_scope('output'):
    weights = common_ops.variable_with_weight_decay(
                  'weight',
                  [512, NUM_CLASSES],
                  stddev=1.0 / math.sqrt(float(512)),
                  useMultiGPU=useMultiGPU)
    biases = common_ops.define_biases(
                 'biases',
                 [NUM_CLASSES],
                 useMultiGPU=useMultiGPU)
    logits = tf.nn.relu(tf.matmul(fc, weights) + biases)
  
  return logits


models_dict = { 'linear_classifier':linear_classifier,
                'ann':ann,
                'cnn':cnn }

def inference(model, images, batch_size,useMultiGPU=False):
  """Use the specified model to infer.
  """
  if model in models_dict:
    print('MODEL : %s' % model)
    if model == 'cnn':
      return models_dict.get(model)(images,
                                     batch_size,
                                     useMultiGPU=useMultiGPU)
    else:
      return models_dict.get(model)(images, useMultiGPU=useMultiGPU)
  else:
    print('%s is invalid model, use default model (linear_classifier) instead.' 
         % model)
    return linear_classifier(images, useMultiGPU=useMultiGPU)
