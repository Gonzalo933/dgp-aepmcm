import numpy as np
import tensorflow as tf

from dgp_aepmcm.nodes.output_node_base import OutputNodeBase


class OutputNodeRegression(OutputNodeBase):
    def __init__(
        self, y_train_tf, y_test_tf, y_train_mean_tf, y_train_std_tf, n_samples, dtype=np.float32
    ):
        OutputNodeBase.__init__(self, y_train_tf, y_test_tf, n_samples)
        self.y_train_mean_tf = y_train_mean_tf
        self.y_train_std_tf = y_train_std_tf
        self.dtype = dtype

    """
    This class represents an output node that will contain the targets for a regression problem
    there is no noise (as it has been added in the last noise layer).
    """

    def set_nodes_as_input(self, list_of_input_nodes):
        assert (
            len(list_of_input_nodes) == 1
        ), "The number of nodes in the last hidden layer should be 1"

        self.list_of_input_nodes = list_of_input_nodes

    def get_logz(self):
        # assert self.n_samples_to_propagate is not None, "Network must be set for training or prediction first"
        input_means, input_vars = self.get_input()
        # This is the last step for calculating logz
        # \int q(y | h^{L-1}) q(h^{L-1}) dh^{L-1} where the distributions q are just the original
        # p distributions (p(y| h^{L-1}, u^L ) and p(h^{L-1}| u^{L-1}))
        # with the inducing points marginalized
        # Log of Gaussian mixture over samples: log Sum_s N(y | means, vars) - log S
        # (N,1)
        log_num_samples = tf.log(tf.cast(tf.shape(input_means)[0], self.dtype))
        return (
            tf.reduce_logsumexp(
                -0.5 * tf.log(2 * np.pi * input_vars)
                - 0.5 * (self.y_train_tf - input_means) ** 2 / input_vars,
                0,
            )
            - log_num_samples
        )

    def get_predicted_values(self):
        # Calculate the predictive distribution for a given x.
        # Return the mean and variance of that distribution (Gaussian mixture)
        input_means, input_vars = self.get_input()  # (S,N,1), (S,N,1)
        output_means = tf.reduce_mean(input_means, 0)  # (N,1)
        # The output variance is the second moment of a Gaussian mixture
        output_vars = tf.reduce_mean(input_means ** 2 + input_vars, 0) - output_means ** 2
        return output_means, output_vars

    def calculate_loglikehood_rmse(self):
        # Expecting not normalized self.y_test_tf
        # np.mean(logsumexp(norm.logpdf(self.y_test_tf, loc=m, scale = np.sqrt(v)), 0) - np.log(S))
        input_means, input_vars = self.get_input()
        log_num_samples = tf.log(tf.cast(tf.shape(input_means)[0], self.dtype))
        norm = tf.distributions.Normal(
            loc=input_means * self.y_train_std_tf + self.y_train_mean_tf,
            scale=tf.sqrt(input_vars) * self.y_train_std_tf,
        )

        # (S, N, 1)
        logpdf = norm.log_prob(tf.cast(self.y_test_tf, self.dtype))
        # RMSE
        mean = (
            tf.reduce_mean(input_means, 0) * self.y_train_std_tf + self.y_train_mean_tf
        )  # mean over the samples
        sq_diff = (mean - self.y_test_tf) ** 2
        # logsumexp over samples
        return (tf.reduce_logsumexp(logpdf, 0) - log_num_samples), sq_diff

    def sample_from_predictive_distribution(self, samples_per_point=1):
        # TODO: Maybe that samples_per_point do actually something? Use placeholder
        input_means, input_vars = self.get_input()  # S,N,D
        S = tf.shape(input_means)[0]
        N = tf.shape(input_means)[1]
        D = tf.shape(input_means)[2]
        # Sample from mixture of gaussians
        # 1. sample from categorical dist

        probs = (1 / tf.cast(S, self.dtype)) * tf.ones(shape=[S], dtype=self.dtype)
        cat = tf.distributions.Categorical(probs=probs)
        samples_categorical = cat.sample(sample_shape=[N])
        # Get corresponding means and vars
        indexes = tf.concat([samples_categorical[:, None], tf.range(0, N)[:, None]], 1)
        mixture_means = tf.gather_nd(
            input_means, indexes
        )  # mixture_means = input_means[samples_categorical, tf.range(0, N), :]
        mixture_vars = tf.gather_nd(input_vars, indexes)  # Size: N,D
        return tf.random_normal(
            shape=[N, D], mean=mixture_means, stddev=tf.sqrt(mixture_vars), dtype=self.dtype
        )

    def get_predictive_distribution_fixed_x(self, y_values):
        input_means, input_vars = self.get_input()  # S,N,D
        # pdf = 1/(2*np.pi*input_vars)**0.5 * tf.exp(-0.5*(y_values - input_means)**2/input_vars)
        norm = tf.distributions.Normal(loc=input_means, scale=tf.sqrt(input_vars))
        return tf.reduce_mean(norm.prob(tf.cast(y_values, self.dtype)), 0)
