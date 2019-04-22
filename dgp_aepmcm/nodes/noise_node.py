##
# This class represents a noisy node in the network
#
import numpy as np
import tensorflow as tf

from dgp_aepmcm.nodes.base_node import BaseNode


class NoiseNode(BaseNode):
    def __init__(self, dtype, initial_value):
        BaseNode.__init__(self)
        self.input_samples = None
        self.output_samples = None
        self.n_samples = None  # Not used in Noisy node
        self.dtype = dtype
        self.initial_value = initial_value
        self.lvar_noise = tf.Variable(
            initial_value=np.log(self.initial_value), name="lvar_noise", dtype=self.dtype
        )

    def calculate_output(self, input_means, input_vars):
        self.input_means = input_means
        self.input_vars = input_vars
        self.output_means = input_means
        self.output_vars = input_vars + tf.exp(self.lvar_noise)

    def get_params(self):
        return [self.lvar_noise]
