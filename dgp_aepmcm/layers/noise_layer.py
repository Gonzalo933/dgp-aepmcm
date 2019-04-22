import numpy as np
import tensorflow as tf

from dgp_aepmcm.layers.base_layer import BaseLayer
from dgp_aepmcm.nodes.noise_node import NoiseNode


class NoiseLayer(BaseLayer):
    def __init__(self, dtype, initial_value):
        BaseLayer.__init__(self)
        self.dtype = dtype
        self.initialized = False
        self.initial_value = initial_value

    def stack_on_previous_layer(self, previous_layer):
        """ Stacks a noise node in each of the nodes of the previous layer.

        That means that each GP node has its own noise at the output

        Args:
            previous_layer (Object): Layer to connect with
        """
        nodes_previous_layer = previous_layer.get_node_list()
        self.n_nodes = len(nodes_previous_layer)

        for gp_node in nodes_previous_layer:
            noise_node = NoiseNode(self.dtype, self.initial_value)
            noise_node.set_nodes_as_input([gp_node])
            self.add_node(noise_node)
        self.stacked = True
