import tensorflow as tf

def lgm_logits(feat, num_classes, labels=None, alpha=0.1, lambda_=0.01):
  '''
  The 3 input hyper-params are explained in the paper.\n
  Support 2 modes: Train, Validation\n
  (1)Train:\n
  return logits, likelihood_reg_loss\n
  (2)Validation:\n
  Set labels=None\n
  return logits\n
  '''
  N = feat.get_shape().as_list()[0]
  feat_len = feat.get_shape()[1]
  means = tf.get_variable('rbf_centers', [num_classes, feat_len], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())

  XY = tf.matmul(feat, means, transpose_b=True)
  XX = tf.reduce_sum(tf.square(feat), axis=1, keep_dims=True)
  YY = tf.reduce_sum(tf.square(tf.transpose(means)), axis=0, keep_dims=True)
  neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

  if labels is None:
    # Validation mode
    psudo_labels = tf.argmax(neg_sqr_dist, axis=1)
    means_batch = tf.gather(means, psudo_labels)
    likelihood_reg_loss = lambda_ * tf.nn.l2_loss(feat - means_batch, name='likelihood_regularization') * (1. / N)
    # In fact, in validation mode, we only need to output neg_sqr_dist.
    # The likelihood_reg_loss and means are only for research purposes.
    return neg_sqr_dist, likelihood_reg_loss, means
  # *(1 + alpha)
  ALPHA = tf.one_hot(labels, num_classes, on_value=alpha, dtype=tf.float32)
  K = ALPHA + tf.ones([N, num_classes], dtype=tf.float32)
  logits_with_margin = tf.multiply(neg_sqr_dist, K)
  # likelihood regularization
  means_batch = tf.gather(means, labels)
  likelihood_reg_loss = lambda_ * tf.nn.l2_loss(feat - means_batch, name='center_regularization') * (1. / N)
  print('LGM loss built with alpha=%f, lambda=%f\n' %(alpha, lambda_))
  return logits_with_margin, likelihood_reg_loss, means


def inference_lgm(image, resnet_size, is_training, num_classes=100, labels=None, reuse=False, output_feat=False):
  print('built lgm inference')
  _, feat = inference(image, resnet_size, is_training, num_classes=num_classes, reuse=reuse, output_feat=True)
  with tf.variable_scope('rbf_loss', reuse=reuse):
      logits, likelihood_reg_loss, means = lgm_logits(feat, num_classes, labels=labels, alpha=0.1, lambda_=0.01)
  if output_feat:
    return logits, likelihood_reg_loss, means, feat
  else:
    return logits, likelihood_reg_loss, means