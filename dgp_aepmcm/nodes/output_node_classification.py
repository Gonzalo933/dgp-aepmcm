##
# This class represents an output node that will contain the targets for a classification problem
# It plays the role of a probit function. The lables are either 1 or -1
#
import numpy as np
import tensorflow as tf
from tensorflow.distributions import Normal

from dgp_aepmcm.nodes.output_node_base import OutputNodeBase


class OutputNodeClassification(OutputNodeBase):
    def __init__(self, y_train_tf, y_test_tf, n_samples, variance_bias, dtype):
        OutputNodeBase.__init__(self, y_train_tf, y_test_tf, n_samples)
        self.norm = Normal(loc=tf.constant(0.0, dtype), scale=tf.constant(1.0, dtype))
        self.dtype = dtype
        self.variance_bias = variance_bias

    def set_nodes_as_input(self, list_of_input_nodes):
        # Override from BaseNode
        assert (
            len(list_of_input_nodes) == 1
        ), "The last node should have just one input, currently {}".format(
            len(list_of_input_nodes)
        )

        self.list_of_input_nodes = list_of_input_nodes

    def get_logz(self):
        # Using step function as likelihood with noise is equivalent to Gaussian cdf.
        # Calculate:
        # log E[ p(y | f^L)] = log \int p(y | f^L) q^\cavity(f^L | x)
        # = log \int p(y | f^L) N(f^L | input_mean, input_vars)
        # with p(y | f^L) a step function (f^L bigger or equal to 0 classified as y=1)
        # By samples:
        # log 1/S \sum_{s=1}^S \Phi((y_{i,s} * input_means) / \sqrt(input_vars))
        # Using log cdf for robustness
        input_means, input_vars = self.get_input()
        S = tf.shape(input_means)[0]

        # Parameter of the log cdf
        alpha = (
            tf.cast(self.y_train_tf, self.dtype)
            * input_means
            / tf.sqrt(input_vars + self.variance_bias)
        )
        return tf.reduce_logsumexp(self.norm.log_cdf(alpha), 0) - tf.log(
            tf.cast(S, self.dtype)
        )

    def calculate_log_likelihood(self):
        # The only difference with the function above is that
        # the means and vars should be calculated using the psoterior instead of the cavity
        # and y_test should be used.
        input_means, input_vars = self.get_input()
        S = tf.shape(input_means)[0]

        # Parameter of the log cdf
        alpha = (
            tf.cast(self.y_test_tf, self.dtype)
            * input_means
            / tf.sqrt(input_vars + self.variance_bias)
        )

        return tf.reduce_logsumexp(self.norm.log_cdf(alpha), 0) - tf.log(
            tf.cast(S, self.dtype)
        )

    def get_predicted_values(self):
        input_means, input_vars = self.get_input()  # S, N, 1
        S = tf.shape(input_means)[0]

        # (S, N, 1)
        alpha = input_means / tf.sqrt(input_vars + self.variance_bias)
        # (N, 1)
        prob = tf.exp(
            tf.reduce_logsumexp(self.norm.log_cdf(alpha), 0)
            - tf.log(tf.cast(S, self.dtype))
        )
        # label[n] = -1 if input_means[n] < 0  else 1
        labels = tf.where(
            tf.less(tf.reduce_sum(input_means, 0), tf.zeros_like(prob)),
            -1 * tf.ones_like(prob),
            tf.ones_like(prob),
        )

        return labels, prob

    def sample_from_latent(self):
        input_means, input_vars = self.get_input()  # S, N, 1
        # Returns samples from H^L
        return tf.random_normal(
            tf.shape(self.input_means),
            mean=self.input_means,
            stddev=tf.sqrt(self.input_vars),
            seed=3,
            dtype=self.dtype,
        )  # seed=3
