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
import diff_models

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'cnn',
                           """Choose which model to use.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of samples to process in a batch.""")
tf.app.flags.DEFINE_integer('max_train_epoch', 10,
                            """Maximum number of training epoch.""")

# Distribute the number of examples for training and evaluation.
NUM_EXAMPLES_FOR_TRAIN = 55000
NUM_EXAMPLES_FOR_EVAL  = 5000


def main(argv=None):
  images, labels = read_data.generate_batch(FLAGS.model, FLAGS.batch_size)
  
  logits = diff_models.inference(FLAGS.model, images, FLAGS.batch_size)
  
  loss = common_ops.cross_entropy_loss(logits, labels)
  
  global_step = tf.Variable(0, trainable=False)
  train_op = common_ops.optimizer(FLAGS.batch_size, global_step).minimize(loss)
  
  accuracy = common_ops.evaluate(logits, labels)

  save_path = os.path.join(os.path.abspath('.'), 'checkpoint')
  checkpoint_path = os.path.join(save_path, 'model.ckpt')
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  saver = tf.train.Saver(tf.all_variables())
  
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    train_steps_per_epoch = NUM_EXAMPLES_FOR_TRAIN // FLAGS.batch_size
    eval_steps_per_epoch  = NUM_EXAMPLES_FOR_EVAL  // FLAGS.batch_size
    for epoch in xrange(FLAGS.max_train_epoch):
      print('epoch %d -----------------------' % epoch)
      
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
