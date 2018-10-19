import math
import torch
import torch.nn as nn
import torchvision
import numpy as np

class LGMLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha_=0.1, lambda_=0.01):
        super(LGMLoss, self).__init__()
        self.num_classes = num_classes
        self._feature_dim = feat_dim
        self.alpha = alpha_
        self.lambda_ = lambda_
        self.means = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.means)
        self.vars = nn.Parameter(torch.ones(num_classes, feat_dim))

    def _classification_probability(self, x, y, mean, var):
        batch_size = x.size()[0]
        print()
        print('x.shape: {}, y.shape: {}'.format(x.shape, y.shape))
        print()
        print('classification_probability')
        print('mean.shape: {}, var.shape: {}'.format(mean.shape, var.shape))


        reshape_var = var.view(-1, 1, self._feature_dim)
        reshape_mean = mean.view(-1, 1, self._feature_dim)
        print()
        print('reshape_mean.shape: {}, reshape_var.shape: {}'.format(
            reshape_mean.shape, reshape_var.shape))


        expand_data = x.unsqueeze_(0)
        print()
        print('expand_data.shape: {}'.format(expand_data.shape))
        #this is extra
        expand_data = torch.transpose(expand_data, 0,1)
        print()
        print('expand_data_transpose.shape: {}'.format(expand_data.shape))

        data_min_mean = expand_data - reshape_mean
        print()
        print('data_min_mean.shape: {}'.format(data_min_mean.shape))


        transpose_data_min_mean = torch.transpose(data_min_mean, 0, 1)
        pair_m_distance = torch.bmm(data_min_mean/(reshape_var+1e-8),
                                    transpose_data_min_mean)/2
        print()
        print('pair_m_distance.shape: {}'.format(pair_m_distance.shape))


        index = torch.from_numpy(np.array([i for i in range(batch_size)]))
        print()
        print('index.shape: {}'.format(index))
        m_distance = torch.transpose(pair_m_distance[:, index, index], 0, 1)
        print()
        print('m_distance.shape: {}'.format(m_distance.shape))
        det = torch.dot(var, 1)
        print()
        print('det.shape: {}'.format(det.shape))
        print()
        label_onehot = torch.index_select(self.means, dim=0, index=y)
        print()
        print('label_onehot.shape: {}'.format(label_onehot.shape))
        adjust_m_distance = m_distance + label_onehot * self.alpha * m_distance
        print('adjust_m_distance.shape: {}'.format(adjust_m_distance.shape))
        probability = torch.exp(-adjust_m_distance)/(det ** 2 + 1e-8)
        print('probability.shape: {}'.format(probability.shape))
        return probability, m_distance

    def _classification_loss(self, probability, y):
        label_onehot = torch.index_select(self.means, dim=0, index=y)
        class_probability = torch.sum(label_onehot*probability, 1)
        print('class_probability.shape: {}'.format(class_probability.shape))
        classification_loss = -torch.log(class_probability / (torch.sum(probability, 1)+1e-8)+1e-8)
        print('classification_loss.shape: {}'.format(classification_loss.shape))
        return classification_loss

    def forward(self, feat, labels=None):
        probability, m_distance = self._classification_probability(feat, labels,
                                                                   self.means, self.vars)
        classification_loss = self._classification_loss(probability, labels)
        return classification_loss, self.means, self.vars


class Mean_LGMLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.1, lambda_=0.01):
        super(LGMLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.lambda_ = lambda_
        self.means = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.means, gain=math.sqrt(2.0))

    def forward(self, feat, labels=None):
        batch_size= feat.size()[0]

        XY = torch.matmul(feat, torch.transpose(self.means, 0, 1))
        XX = torch.sum(feat ** 2, dim=1, keepdim=True)
        YY = torch.sum(torch.transpose(self.means, 0, 1)**2, dim=0, keepdim=True)
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)
        """
        if labels is None:
            psudo_labels = torch.argmax(neg_sqr_dist, dim=1)
            means_batch = torch.index_select(self.means, dim=0, index=psudo_labels)
            likelihood_reg_loss = self.lambda_ * (torch.sum((feat - means_batch)**2) / 2) * (1. / batch_size)
            return neg_sqr_dist, likelihood_reg_loss, self.means
"""
        labels_reshped = labels.view(labels.size()[0], -1)

        if torch.cuda.is_available():
            ALPHA = torch.zeros(batch_size, self.num_classes).cuda().scatter_(1, labels_reshped, self.alpha)
            K = ALPHA + torch.ones([batch_size, self.num_classes]).cuda()
        else:
            ALPHA = torch.zeros(batch_size, self.num_classes).scatter_(1, labels_reshped, self.alpha)
            K = ALPHA + torch.ones([batch_size, self.num_classes])

        logits_with_margin = torch.mul(neg_sqr_dist, K)
        #print("logits_with_margin: {}, neg_sqr_dist: {}, K: {}".format(logits_with_margin, neg_sqr_dist, K))
        """
        y_hat_softmax = torch.argmax(torch.nn.functional.softmax(logits_with_margin, 1), 1)
        #print(y_hat_softmax.shape, y_true.shape)
        log = torch.log(y_hat_softmax.float())
        sum = -torch.sum(labels * log.long())
        logits_with_margin = torch.mean(sum.float())
"""

        #print('logits_with_margin.shape ',logits_with_margin.shape)
        means_batch = torch.index_select(self.means, dim=0, index=labels)
        likelihood_reg_loss = self.lambda_ * (torch.sum((feat - means_batch)**2) / 2) * (1. / batch_size)
        #print('likelihood_reg_loss: {}'.format(likelihood_reg_loss))
        return logits_with_margin, likelihood_reg_loss, self.means
