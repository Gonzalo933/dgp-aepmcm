##
# This file contains the object that represents an classification output layer to the network
#

import numpy as np

from dgp_aepmcm.layers.output_layer_base import OutputLayerBase
from dgp_aepmcm.nodes.output_node_classification import OutputNodeClassification
from dgp_aepmcm.nodes.output_node_multiclass import OutputNodeMulticlass


class OutputLayerClassification(OutputLayerBase):
    def __init__(
        self,
        y_train_tf,
        y_test_tf,
        n_samples,
        n_classes,
        variance_bias,
        include_noise_in_labels,
        noise_in_labels_trainable,
        dtype,
    ):
        OutputLayerBase.__init__(self, y_test_tf)
        self.n_classes = n_classes
        if self.n_classes == 2:
            output_node = OutputNodeClassification(
                y_train_tf, y_test_tf, n_samples, variance_bias, dtype
            )
        else:
            output_node = OutputNodeMulticlass(
                y_train_tf,
                y_test_tf,
                n_samples,
                n_classes,
                include_noise_in_labels,
                noise_in_labels_trainable,
                dtype,
            )
        self.add_node(output_node)

    def calculate_log_likelihood(self):
        return self.get_node_list()[0].calculate_log_likelihood()

    def sample_from_latent(self):
        return self.get_node_list()[0].sample_from_latent()
