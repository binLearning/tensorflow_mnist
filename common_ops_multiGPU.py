"""common operations for using multiple GPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from six.moves import xrange

import tensorflow as tf

import read_data
import common_ops
import diff_models


def tower_loss(model, batch_size, scope):
  """Calculate the total loss on a single tower.
     This function constructs the entire model
     but shares the variables across all towers.
  """
  images, labels = read_data.generate_batch(model, batch_size)
  
  logits = diff_models.inference(model, images, batch_size, useMultiGPU=True)
  
  _ = common_ops.cross_entropy_loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')
  
  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
     Note that this is the synchronization point across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
