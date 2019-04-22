##
# This file contains the object that represents an classification output layer to the network
#
import tensorflow as tf

from dgp_aepmcm.layers.base_layer import BaseLayer


class OutputLayerBase(BaseLayer):
    def __init__(self, y_test_tf):
        BaseLayer.__init__(self)
        self.y_test_tf = y_test_tf

    def get_layer_contribution_to_energy(self):
        return tf.reduce_sum(self.get_node_list()[0].get_logz(), axis=None)

    def get_predicted_values(self):
        return self.get_node_list()[0].get_predicted_values()

    def calculate_loglikehood_rmse(self):
        pass

    def calculate_log_likelihood(self):
        pass
