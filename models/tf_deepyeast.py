from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout
import tensorflow as tf

def preprocess_input(x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def batch_norm_relu(input):
    inputs = tf.layers.batch_normalization(
        inputs=input, axis=3, momentum=_BATCH_NORM_EPSILON, center=True,
        scale=True, fused=True
    )
    inputs = tf.nn.relu(inputs)
    return inputs


def fixed_padding(inputs, kernel_size):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.contrib.layers.xavier_initializer())


class DeepYeast:
    def __init__(self):
        print('start building models')

    def build(self, input):
        assert input.get_shape().as_list()[1:] == [64, 64, 2]
        with tf.variable_scope("block_one"):
            # Block 1
            x = self.conv_layer(input, 64, 'conv1')
            x = self.conv_layer(x, 64, 'conv2')
            x = self.max_pool(x, 'pool')

        with tf.variable_scope("block_two"):
            # Block 2
            x = self.conv_layer(x, 128, 'conv1')
            x = self.conv_layer(x, 128, 'conv2')
            x = self.max_pool(x, 'pool')

        with tf.variable_scope("block_three"):
            # Block 3
            x = self.conv_layer(x, 256, 'conv1')
            x = self.conv_layer(x, 256, 'conv2')
            x = self.conv_layer(x, 256, 'conv3')
            x = self.conv_layer(x, 256, 'conv4')
            x = self.max_pool(x, 'pool')

        with tf.variable_scope("cls_block"):
            # Classification block
            x = Flatten(name='flatten')(x)
            x = tf.layers.dense(x, 512, name='dense1')
            x = self.batch_norm(x, 'bn1')
            x = tf.nn.relu(x, name='relu1')
            x = tf.layers.dropout(x, rate=0.5, name='dropout1')
            x = tf.layers.dense(x, 512, name='dense2')
            x = self.batch_norm(x, 'bn2')
            x = tf.nn.relu(x, name='relu2')
            x = tf.layers.dropout(x, rate=0.5, name='dropout2')
            x = tf.layers.dense(x, 512, name='dense_label')
            self.prob = tf.nn.softmax(x, name="prob")

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, input, filter, name):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(input, filter, kernel_size=3, padding='SAME')
            bn_relu = batch_norm_relu(conv)
            return bn_relu

    def batch_norm(self, input, name):
        inputs = tf.layers.batch_normalization(
            inputs=input, momentum=_BATCH_NORM_EPSILON, name=name
        )
        return inputs

    def fc_layer(self, bottom, units):
        fc = tf.layers.dense(bottom, units)
        return fc