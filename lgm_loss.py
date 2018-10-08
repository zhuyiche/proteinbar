import keras.backend as K
import keras
HALF_LOG_TWOPI = 0.91893853320467267  # (1/2)*log(2*pi)


def lgm_loss(num_classes=12, alpha=0.1, lambda_=0.01):
    """Build log-likelihood loss for Gaussian Mixture Densities.
    Args:
        c (int): Number of output dimensions.
        m (int): Number of gaussians in the mixture.
    Returns:
        Loss function.
    """

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