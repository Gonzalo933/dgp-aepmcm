##
# This class represents an output node that will contain the targets for a multi class problem
# This node must be conected two two other nodes and has the form Prod_k!=y Theta(f_y(x) - f_k(x))
# LogZ is approximated by a quadrature technique, with fixed points sampled from a standard gaussian
# The potential labels go from 0 to n_classes - 1
#
import math

import numpy as np

import tensorflow as tf
from dgp_aepmcm.nodes.output_node_base import OutputNodeBase


class OutputNodeMulticlass(OutputNodeBase):
    def __init__(
        self,
        y_train_tf,
        y_test_tf,
        n_samples,
        n_classes,
        noise_in_labels,
        noise_in_labels_trainable,
        dtype,
    ):
        assert (
            n_classes > 2
        ), "The number of classes should be greater than 2 to use the layer Multiclass"
        # TODO: n_classes = D  so we don't really need the parameter
        OutputNodeBase.__init__(self, y_train_tf, y_test_tf, n_samples)
        self.n_classes = n_classes
        self.grid_size = 100
        self.dtype = dtype
        self.norm = tf.distributions.Normal(
            loc=tf.constant(0.0, self.dtype), scale=tf.constant(1.0, self.dtype)
        )
        # Grid for Gauss–Hermite quadrature
        self.grid = tf.get_variable(
            "grid",
            [self.grid_size],
            initializer=tf.constant_initializer(np.linspace(-7, 7, self.grid_size)),
            dtype=self.dtype,
        )
        self.noise_in_labels_trainable = noise_in_labels_trainable
        self.latent_prob_wrong_label = None
        if noise_in_labels:
            # Latent variable that represents noise in the labels
            self.latent_prob_wrong_label = tf.get_variable(
                name="lpwl",
                shape=[1],
                initializer=tf.constant_initializer(self.logit(1e-3), dtype=self.dtype),
                dtype=self.dtype,
            )

    def set_nodes_as_input(self, list_of_input_nodes):
        # Overrided from BaseNode
        assert (
            len(list_of_input_nodes) == self.n_classes
        ), "The number of nodes in the last (GP) layer should be equal to the number of clases"

        self.list_of_input_nodes = list_of_input_nodes

    def get_params(self):
        if self.latent_prob_wrong_label is not None and self.noise_in_labels_trainable:
            return [self.latent_prob_wrong_label]
        return []

    @classmethod
    def logit(self, p):
        return np.log(p / (1.0 - p))

    def get_logz(self, use_y_training=True, target_class=None):
        # Mean and the variance of the function corresponding to the observed class
        # (S, N, K)  K -> n_classes
        input_means, input_vars = self.get_input()

        if target_class is not None:
            # We want to calculate p(y=target_class | f^L)
            N = tf.shape(input_means)[1]
            targets = tf.ones(shape=(N,), dtype=tf.int32) * target_class
        elif use_y_training:
            targets = tf.reshape(self.y_train_tf, (-1,))
        else:
            targets = tf.reshape(self.y_test_tf, (-1,))
        return self.calculate_log_prob_y(input_means, input_vars, targets)

    def calculate_log_likelihood(self):
        # The only difference with the logz calculation
        # the means and vars should be calculated using the psoterior instead of the cavity
        # and y_test should be used instead of y_train.
        return self.get_logz(use_y_training=False, target_class=None)

    def calculate_log_prob_y(self, input_means, input_vars, targets):
        S, N = tf.shape(input_means)[0], tf.shape(input_means)[1]

        # Part of the next code is extracted from GPflow
        # uses gauss hermite quadrature (see wikipedia article)
        # targets = 0, 1, 2, 3, ... K - 1
        # (grid_size, )
        gh_x, gh_w = self.hermgauss(self.grid_size)
        # (S, grid_size, 1)
        gh_w = tf.tile(gh_w[None, :, None], [S, 1, 1])

        # (S, grid_size)
        # gh_x = tf.tile(gh_x[None, :], [S, 1])
        # Targets expressed in a one hot enconding matrix with ones in the position of the class
        # (S, N, K)
        targets_one_hot_on = tf.one_hot(
            targets,
            self.n_classes,
            tf.constant(1.0, self.dtype),
            tf.constant(0.0, self.dtype),
            dtype=self.dtype,
        )
        # Only select the means or vars corresponding to the dimension
        #  of the class that we are interested (reduce over K)
        # (S, N, 1)
        means_class_y_selected = tf.reduce_sum(
            targets_one_hot_on * input_means, -1, keepdims=True
        )
        # (S, N, 1)
        vars_class_y_selected = tf.reduce_sum(
            targets_one_hot_on * input_vars, -1, keepdims=True
        )

        # As we have to do a change of variable for the Gauss-Hermite quadrature
        # we calculate all the points to evaluate the quadrature.
        # (S, N, grid_size)
        X = means_class_y_selected + gh_x * tf.sqrt(
            tf.clip_by_value(2.0 * vars_class_y_selected, 1e-10, np.inf)
        )
        # tf.expand_dims(X, 2) -> (S, N, 1, grid_size)
        # tf.expand_dims(input_means, 3) -> (S, N, K, 1)
        # (tf.expand_dims(X, 2) - tf.expand_dims(input_means, 3)) -> (S, N, K, grid_size)
        # dist -> (S, N, K, grid_size)
        dist = (tf.expand_dims(X, 2) - tf.expand_dims(input_means, 3)) / tf.expand_dims(
            tf.sqrt(tf.clip_by_value(input_vars, 1e-10, np.inf)), 3
        )
        # (S, N, K, grid_size)
        cdfs = self.norm.cdf(dist)
        # (S, N)
        # One in positions different to targets (logical not of targets_one_hot_on)
        oh_off = tf.cast(tf.one_hot(targets, self.n_classes, 0.0, 1.0), self.dtype)
        oh_off_tiled_by_samples = tf.tile(oh_off[None, :, :], [S, 1, 1])

        # Blank out all the distances on the selected latent function
        # (S, N, K, grid_size)
        cdfs = cdfs * (1 - 2e-4) + 1e-4
        cdfs = cdfs * tf.expand_dims(oh_off_tiled_by_samples, 3) + tf.expand_dims(
            tf.tile(targets_one_hot_on[None, :, :], [S, 1, 1]), 3
        )
        # Reduce over the classes. (product of k not equal to y_i)
        cdfs_reduced = tf.reduce_prod(cdfs, 2)
        # If there is noise in labels, the likelihood is not
        # just 1 if correct classified or 0 if not
        # when self.latent_prob_wrong_label is 1 that means that the label for y_i is wrong
        # and we assign equal prob. for the rest of the classes.
        if self.latent_prob_wrong_label is not None:
            cdfs_reduced = cdfs_reduced * tf.sigmoid(
                tf.cast(-1.0 * self.latent_prob_wrong_label, self.dtype)
            ) + (1 - cdfs_reduced) * tf.sigmoid(self.latent_prob_wrong_label) / (
                self.n_classes - 1
            )

        # reduce_sum over samples
        # Final result -> (N, 1)
        probs = tf.reduce_sum(cdfs_reduced @ gh_w, 0)
        log_probs = (
            tf.log(probs) - np.log(np.sqrt(np.pi)) - tf.log(tf.cast(S, self.dtype))
        )

        return log_probs

    def hermgauss(self, n: int):
        # This has been extracted from GP flow. Return the locations and weights of GH quadrature
        x, w = np.polynomial.hermite.hermgauss(n)
        return x.astype(self.dtype), w.astype(self.dtype)

    def get_predicted_values(self):
        # input_means, input_vars = self.get_input()
        # The classification rule is:
        # y_i = arg max_k  f_k(x_i)
        # That means that y_i is assigned the index of the latent function with higher value

        probs = []
        for target_class in range(self.n_classes):
            probs.append(tf.exp(self.get_logz(target_class=target_class)[:, 0]))

        # Returns labels and confidence on prediction.
        # sum(probs_i) should be 1
        prob = tf.stack(probs, axis=1)
        labels = tf.argmax(prob, 1)[:, None]

        return labels, prob
