"""Parses the MNIST binary file to read image and label data."""

from __future__ import absolute_import
from __future__ import print_function

import os
from six.moves import xrange

import tensorflow as tf

# Global constants describing the MNIST data set.
IMAGE_SIZE     = 28
IMAGE_CHANNELS = 1
NUM_PIXELS     = IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNELS
NUM_CLASSES    = 10

NUM_EXAMPLES_IN_TRAIN_SET = 60000
NUM_EXAMPLES_IN_TEST_SET  = 10000

FOLDER_PATH = r"/root/binLearning/database/MNIST/"


def _read_images(test_data=False, as_image=True, for_show=False):
  """Reads and parses the binary file which contains training/test images.
  """
  if not test_data:
    filename = os.path.join(FOLDER_PATH, 'train-images.idx3-ubyte')
  else:
    filename = os.path.join(FOLDER_PATH, 't10k-images.idx3-ubyte')
  if not os.path.exists(filename):
    raise ValueError('The file dose not exist.')
  
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer([filename])
  
  # The first 16 bytes contain file information:
  # [offset]     [type]             [value]              [description]
  # 0000         32 bit integer     0x00000803(2051)     magic number
  # 0004         32 bit integer     60000/10000          number of images 
  # 0008         32 bit integer     28                   number of rows
  # 0012         32 bit integer     28                   number of columns
  # ...(pixel value)
  header_bytes = 16
  # Every record consists of an image, with a fixed number of bytes for each.
  record_bytes = IMAGE_SIZE * IMAGE_SIZE
  
  # Create a FixedLengthRecordReader to read record.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes,
                                      header_bytes=header_bytes)
  _, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8.
  image = tf.decode_raw(value, tf.uint8)
  
  if for_show:
    reshape_image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE])
    return reshape_image

  if as_image: # for CNN
    # Reshape from [height * width * channels] to [height, width, channels].
    reshape_image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])

    # Subtract off the mean and divide by the variance of the pixels.
    # Linearly scales image to have zero mean and unit norm.
    preproc_image = tf.image.per_image_whitening(reshape_image)
  else: # for linear classifier / ANN
    # To avoid ValueError: All shapes must be fully defined:...
    image.set_shape([IMAGE_SIZE * IMAGE_SIZE])
    
    # Cast image pixel value from tf.uint8 to tf.float32
    float_image = tf.cast(image, tf.float32)
    
    # normalization
    preproc_image = tf.div(float_image, 255.0)

  return preproc_image


def _read_labels(test_data=False):
  """Reads and parses the binary file which contains training/test labels.
  """
  if not test_data:
    filename = os.path.join(FOLDER_PATH, 'train-labels.idx1-ubyte')
  else:
    filename = os.path.join(FOLDER_PATH, 't10k-labels.idx1-ubyte')
  if not os.path.exists(filename):
    raise ValueError('The file dose not exist.')
  
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer([filename])
  
  # The first 8 bytes contain file information:
  # [offset]     [type]             [value]              [description]
  # 0000         32 bit integer     0x00000801(2049)     magic number
  # 0004         32 bit integer     60000/10000          number of items 
  # ...(label value)
  header_bytes = 8
  # Every record consists of a label, with a fixed number of bytes for each.
  record_bytes = 1
  
  # Create a FixedLengthRecordReader to read record.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes,
                                      header_bytes=header_bytes)
  _, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8, then cast to int32.
  record = tf.cast(tf.decode_raw(value, tf.uint8), tf.int32)
  
  # Reshape from [1] to a scalar shape [].
  label = tf.reshape(record, [])

  return label


def generate_batch(model, batch_size, test_data=False):
  """Construct a queued batch of images and labels.
  """
  if model == 'cnn':
    as_image = True
  else:
    as_image = False

  image = _read_images(test_data=test_data, as_image=as_image)
  label = _read_labels(test_data=test_data)

  images_batch, labels_batch = tf.train.batch([image, label],
                                              batch_size  = batch_size,
                                              num_threads = 1,
                                              capacity    = batch_size * 8)

  return images_batch, tf.reshape(labels_batch, [batch_size])


"""
def func():
  image,label = generate_batch(...)

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    # run training step or whatever
    
    coord.request_stop()
    coord.join(threads)
"""
