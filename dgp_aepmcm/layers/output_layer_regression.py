import numpy as np

from dgp_aepmcm.layers.output_layer_base import OutputLayerBase
from dgp_aepmcm.nodes.output_node_regression import OutputNodeRegression


class OutputLayerRegression(OutputLayerBase):
    def __init__(
        self, y_train_tf, y_test_tf, y_train_mean_tf, y_train_std_tf, n_samples, dtype=np.float32
    ):
        OutputLayerBase.__init__(self, y_test_tf)
        self.n_samples = n_samples
        self.n_nodes = 1
        output_node = OutputNodeRegression(
            y_train_tf, y_test_tf, y_train_mean_tf, y_train_std_tf, n_samples, dtype
        )
        self.add_node(output_node)

    def sample_from_predictive_distribution(self, samples_per_point=1):
        return self.get_node_list()[0].sample_from_predictive_distribution(samples_per_point)

    def get_predictive_distribution_fixed_x(self, y_values):
        return self.get_node_list()[0].get_predictive_distribution_fixed_x(y_values)

    def calculate_loglikehood_rmse(self):
        """ Calculates LL and RMSE for a regression problem

        As the dimension of the Y variables should be 1
        this layer should only have 1 node
        """
        return self.get_node_list()[0].calculate_loglikehood_rmse()
