"""Training on a single device."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange
import time

import tensorflow as tf

import read_data
import common_ops
import common_ops_multiGPU
import diff_models

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'cnn',
                           """Choose which model to use.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of samples to process in a batch.""")
tf.app.flags.DEFINE_integer('max_train_epoch', 10,
                            """Maximum number of training epoch.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")

# Distribute the number of examples for training and evaluation.
NUM_EXAMPLES_FOR_TRAIN = 55000
NUM_EXAMPLES_FOR_EVAL  = 5000


def main(argv=None):
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    global_step = tf.Variable(0, trainable=False)
    optimizer = common_ops.optimizer(FLAGS.batch_size, global_step)
    
    tower_gradients = []
    for num_id in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % num_id):
        with tf.name_scope('tower_%d' % num_id) as scope:
          # Calculate the loss for one tower.
          loss = common_ops_multiGPU.tower_loss(FLAGS.model, 
                                                FLAGS.batch_size, 
                                                scope)
          
          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()
          
          # Calculate the gradients on this tower.
          gradients = optimizer.compute_gradients(loss)
          
          # Keep track of the gradients across all towers.
          tower_gradients.append(gradients)
          
    # Calculate the mean of each gradient.
    avg_gradients = common_ops_multiGPU.average_gradients(tower_gradients)
    
    # Apply the gradients to adjust the shared variables.
    train_op = optimizer.apply_gradients(avg_gradients,
                                         global_step=global_step)
    
    save_path = os.path.join(os.path.abspath('.'), 'checkpoint')
    checkpoint_path = os.path.join(save_path, 'model.ckpt')
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    saver = tf.train.Saver(tf.all_variables())
  
    config=tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
      sess.run(tf.initialize_all_variables())
      
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      
      train_steps_per_epoch = NUM_EXAMPLES_FOR_TRAIN // FLAGS.batch_size
      eval_steps_per_epoch  = NUM_EXAMPLES_FOR_EVAL  // FLAGS.batch_size
      for epoch in xrange(FLAGS.max_train_epoch):
        print('epoch %d -------------------' % epoch)
        
        train_loss_per_epoch = 0.0
        for loops in xrange(train_steps_per_epoch):
          loss_per_batch, _ = sess.run([loss, train_op])
          train_loss_per_epoch += loss_per_batch
        train_loss_per_epoch /= train_steps_per_epoch
        print('training loss   : %f' % train_loss_per_epoch)
        
        eval_loss_per_epoch = 0.0
        for loops in xrange(eval_steps_per_epoch):
          eval_loss_per_epoch += sess.run(loss)
        eval_loss_per_epoch /= eval_steps_per_epoch
        print('evaluation loss : %f' % eval_loss_per_epoch)
        
        if epoch % 5 == 0:
          saver.save(sess, checkpoint_path, global_step=epoch)
      
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
    

if __name__ == '__main__':
  tf.app.run()
