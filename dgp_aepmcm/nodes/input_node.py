import tensorflow as tf

from dgp_aepmcm.nodes.base_node import BaseNode


class InputNode(BaseNode):
    def __init__(self, x_tf, n_samples_dict, network_set_for_training_tf):
        """Instantiates an input node for the network.

        Args:
            x_tf (Tensor): Tensorflow placeholder with x data (training or test) for the network. Size (N,D).
            n_samples_dict (int): Dictionary with number of samples to propagate trough the network.
            network_set_for_training_tf (Tensor): Integer placeholder of shape=() that has the value 1 in case
                that the network is in training state or 0 for prediction state

        """
        BaseNode.__init__(self)
        self.x_tf = x_tf
        self.n_samples_dict = n_samples_dict
        self.network_set_for_training_tf = network_set_for_training_tf
        self.output_means = None
        self.output_vars = None

    def initialize(self):
        pass

    def get_output(self):
        """Calculates the output for an input node which is a tensor
            with the training data replicated n_samples_dict(S) times.
            The shape of this tensor is (S,N,D).
        """
        S = tf.cond(
            tf.equal(self.network_set_for_training_tf, 1.0),
            lambda: self.n_samples_dict["training"],
            lambda: self.n_samples_dict["prediction"],
        )

        self.output_means = tf.tile(self.x_tf[None, :, :], [S, 1, 1])
        self.output_vars = self.output_means * 0.0
        return self.output_means, self.output_vars

    def get_input(self):
        raise Exception("This is an input node, does not receive input from other nodes.")

    def forward_pass_computations(self):
        pass
