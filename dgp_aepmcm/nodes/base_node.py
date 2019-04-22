##
# This class represents a node within the network
#

import numpy as np
import tensorflow as tf


class BaseNode:
    def __init__(self):
        self.list_of_output_nodes = None
        self.list_of_input_nodes = None
        self.input_means = None
        self.input_vars = None
        self.n_samples_to_propagate = None

    def set_nodes_as_input(self, list_of_input_nodes):
        self.list_of_input_nodes = list_of_input_nodes

    def get_list_of_input_nodes(self):
        return self.list_of_input_nodes

    def set_nodes_as_output(self, list_of_output_nodes):
        return self.list_of_output_nodes

    def calculate_output(self, input_means, input_vars):
        self.input_means = input_means
        self.input_vars = input_vars
        self.output_means = input_means
        self.output_vars = input_vars

    def get_input(self):
        assert self.input_means is not None and self.input_vars is not None
        return self.input_means, self.input_vars

    def find_input(self):
        assert self.list_of_input_nodes is not None

        input_means = []
        input_vars = []
        for node in self.list_of_input_nodes:
            input_means_current, input_vars_current = node.get_output()
            input_means.append(input_means_current)
            input_vars.append(input_vars_current)

        return (
            tf.concat(input_means, 2, name="input_means"),
            tf.concat(input_vars, 2, name="input_vars"),
        )

    def get_output(self):
        return self.output_means, self.output_vars

    def get_params(self):
        return []

    def forward_pass_computations(self):
        """ Get the input to the layer and calculate the output """
        input_means, input_vars = self.find_input()
        with tf.control_dependencies([input_means, input_vars]):
            self.calculate_output(input_means, input_vars)

    # This should be overwritten
    def get_energy_contribution(self):
        return 0.0
