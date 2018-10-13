import keras.backend as K
import keras
HALF_LOG_TWOPI = 0.91893853320467267  # (1/2)*log(2*pi)
import tensorflow as tf


def logits_lgm_loss(num_classes=12, alpha=0.1, lambda_=0.01):
    def lgm_logits(y_true, y_pred):
      '''
      The 3 input hyper-params are explained in the paper.\n
      Support 2 modes: Train, Validation\n
      (1)Train:\n
      return logits, likelihood_reg_loss\n
      (2)Validation:\n
      Set labels=None\n
      return logits\n
      '''
      N = y_true.shape().as_list()[0]
      feat_len = y_true.shape()[1]
      means = tf.get_variable('rbf_centers', [num_classes, feat_len], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())

      XY = tf.matmul(y_true, means, transpose_b=True)
      XX = tf.reduce_sum(tf.square(y_true), axis=1, keep_dims=True)
      YY = tf.reduce_sum(tf.square(tf.transpose(means)), axis=0, keep_dims=True)
      neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

      # *(1 + alpha)
      ALPHA = tf.one_hot(y_pred, num_classes, on_value=alpha, dtype=tf.float32)
      K = ALPHA + tf.ones([N, num_classes], dtype=tf.float32)
      logits_with_margin = tf.multiply(neg_sqr_dist, K)
      logits_with_margin_sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_with_margin)
      logits_with_margin_mean = tf.reduce_mean(logits_with_margin_sparse)
      #print('LGM loss built with alpha=%f, lambda=%f\n' %(alpha, lambda_))

      means_batch = tf.gather(means, y_pred)
      likelihood_reg_loss = lambda_ * tf.nn.l2_loss(y_true - means_batch, name='center_regularization') * (1. / N)

      return logits_with_margin_mean + likelihood_reg_loss
    return lgm_logits


def likelihood_lgm_loss(num_classes=12, alpha=0.1, lambda_=0.01):
    def lgm_logits(y_true, y_pred):
      '''
      The 3 input hyper-params are explained in the paper.\n
      Support 2 modes: Train, Validation\n
      (1)Train:\n
      return logits, likelihood_reg_loss\n
      (2)Validation:\n
      Set labels=None\n
      return logits\n
      '''
      N = y_true.shape().as_list()[0]
      feat_len = y_true.shape()[1]
      means = tf.get_variable('rbf_centers', [num_classes, feat_len], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
      # likelihood regularization
      means_batch = tf.gather(means, y_pred)
      likelihood_reg_loss = lambda_ * tf.nn.l2_loss(y_true - means_batch, name='center_regularization') * (1. / N)
      #print('LGM loss built with alpha=%f, lambda=%f\n' %(alpha, lambda_))
      return likelihood_reg_loss
    return lgm_logits
"""
def lgm_loss(num_classes=12, alpha=0.1, lambda_=0.01):

    def lgm_logits(y_true, y_pred):
        '''
        The 3 input hyper-params are explained in the paper.\n
        Support 2 modes: Train, Validation\n
        (1)Train:\n
        return logits, likelihood_reg_loss\n
        (2)Validation:\n
        Set labels=None\n
        return logits\n
        '''
        N = y_true.get_shape().as_list()[0]
        feat_len = y_true.get_shape()[1]
        means = keras.backend.variable([num_classes, feat_len], dtype='float32')

        means_transpose = keras.backend.transpose(means)
        XY = keras.backend.dot(y_true, means_transpose)
        XX = keras.backend.mean(keras.backend.square(y_true), axis=1, keepdims=True)
        YY = keras.backend.mean(keras.backend.square(keras.backend.transpose(means)), axis=0, keepdims=True)
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

        # *(1 + alpha)
        ALPHA = keras.backend.one_hot(y_pred, num_classes)
        K = ALPHA + keras.backend.ones([N, num_classes], dtype='float32')
        logits_with_margin = keras.backend.dot(neg_sqr_dist, K)
        # likelihood regularization
        means_batch = keras.backend.gather(means, y_pred)

        y_true_set = y_true - means_batch
        likelihood_reg_loss = lambda_ * (keras.backend.pow(y_true_set, 2) / 2) * (1. / N)
        print('LGM loss built with alpha=%f, lambda=%f\n' % (alpha, lambda_))
        return logits_with_margin, likelihood_reg_loss, means

    return lgm_logits

def likelihood_lgm_loss(num_classes=12, alpha=0.1, lambda_=0.01):


    def lgm_logits(y_true, y_pred):
        '''
        The 3 input hyper-params are explained in the paper.\n
        Support 2 modes: Train, Validation\n
        (1)Train:\n
        return logits, likelihood_reg_loss\n
        (2)Validation:\n
        Set labels=None\n
        return logits\n
        '''
        N = y_true.get_shape().as_list()[0]
        feat_len = y_true.get_shape()[1]
        means = keras.backend.variable([num_classes, feat_len], dtype='float32')

        means_transpose = keras.backend.transpose(means)
        XY = keras.backend.dot(y_true, means_transpose)
        XX = keras.backend.mean(keras.backend.square(y_true), axis=1, keepdims=True)
        YY = keras.backend.mean(keras.backend.square(keras.backend.transpose(means)), axis=0, keepdims=True)
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

        # *(1 + alpha)
        ALPHA = keras.backend.one_hot(y_pred, num_classes)
        K = ALPHA + keras.backend.ones([N, num_classes], dtype='float32')
        logits_with_margin = keras.backend.dot(neg_sqr_dist, K)
        # likelihood regularization
        means_batch = keras.backend.gather(means, y_pred)

        y_true_set = y_true - means_batch
        likelihood_reg_loss = lambda_ * (keras.backend.pow(y_true_set, 2) / 2) * (1. / N)
        print('LGM loss built with alpha=%f, lambda=%f\n' % (alpha, lambda_))
        return likelihood_reg_loss

    return lgm_logits
"""