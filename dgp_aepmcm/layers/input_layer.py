##
# This file contains the object that represents an input layer to the network
#
from dgp_aepmcm.layers.base_layer import BaseLayer
from dgp_aepmcm.nodes.input_node import InputNode


class InputLayer(BaseLayer):
    def __init__(self, x_train_tf, d_input, n_samples_dict, network_set_for_training_tf):
        """Instantiates an input layer for the network. It has a node that outputs (S,N,D)

        Args:
            x_train_tf (Tensor): Tensorflow placeholder with x_train data for the network. Size (N,D).
            d_input (int): Number of dimensions of x_train.
            n_samples_dict (int): Dictionary with number of samples to propagate trough the network.
            network_set_for_training_tf (Tensor): Integer placeholder of shape=() that has the value 1 in case
                that the network is in training state or 0 for prediction state

        """
        n_nodes = d_input
        BaseLayer.__init__(self)
        self.n_nodes = n_nodes
        self.n_samples_dict = n_samples_dict
        input_node = InputNode(x_train_tf, self.n_samples_dict, network_set_for_training_tf)
        self.add_node(input_node)
        self.initialized = False
