##
# This file contains the object that represents an input layer to the network
#
import numpy as np
import tensorflow as tf

from dgp_aepmcm.layers.base_layer import BaseLayer
from dgp_aepmcm.nodes.gp_node import GPNode


class GPLayer(BaseLayer):

    # n_points is the total number of training points used for cavity computation in GP_node

    def __init__(
        self,
        W,
        n_inducing_points,
        n_points,
        n_nodes,
        input_d,
        first_gp_layer,
        jitter,
        share_z,
        share_kernel_params,
        q_initializations,
        z_initializations=None,
        seed=None,
        dtype=None,
    ):
        """Instantiates a GP node

        Args:
            W (Tensor): Matrix to multiply the inputs to use ass mean function of the GP. m(x)= XW.
            n_inducing_points (int): Number of inducing points to use on the node (M).
            n_points (int): Total number of training points (N).
            n_nodes (int): Number of nodes to add in this GP Layer (D_layer).
            input_d (int): Input dimensions (D_{layer-1}).
            first_gp_layer (Boolean): if this is the first GP layer of the network.
            jitter (float): Jitter level to add to the diagonal of Kxx, bigger jitters improve numerical stability.
            share_z (Boolean): If True all the nodes in the GP layer share the same inducing points
                this makes the model less flexible but faster (hopefully...).
            q_initializations (str): Initializations of the posterior approximation q(u) params. Valid values are:
                'random' (default): Mean and covariance initialized from random normal.
                'deterministic': Mean initialized to mean of the prior p(u) and cov. to 1e-5 * Kzz (1e-5 * prior cov)
                'prior': Mean and cov. initialized to the prior covariance.
            z_initializations (ndarray): If not None, points to initialize the inducing points.
            seed (int): seed to use for random functions.
            dtype (type): Type to use in tf operations.
        """
        assert n_points > 0 and n_nodes > 0 and input_d > 0
        BaseLayer.__init__(self)
        self.n_inducing_points = n_inducing_points
        self.n_nodes = n_nodes
        self.input_d = input_d
        self.share_z = share_z
        self.share_kernel_params = share_kernel_params
        self.z_initializations = z_initializations
        if dtype is None:
            raise ValueError("dtype should be specified (gp_layer)")
        self.dtype = dtype
        self.initialized = False

        self.shared_layer_tf_variables = {}
        # GP layer has to control de inducing points in the case that those are shared between nodes.
        if self.share_z:
            # As all Z are shared between nodes we have to create the variable here
            self.shared_layer_tf_variables["Z"] = tf.get_variable(
                "Z",
                shape=[self.n_inducing_points, self.input_d],
                initializer=tf.zeros_initializer,
                dtype=self.dtype,
            )
        if self.share_kernel_params:
            # Log lengthscale (rbf kernel) ARD enabled
            self.shared_layer_tf_variables["lls"] = tf.get_variable(
                "lls",
                shape=[self.input_d],
                initializer=tf.zeros_initializer,
                dtype=self.dtype,
            )
            # Log scaling factor (sigma, rbf kernel)
            self.shared_layer_tf_variables["lsf"] = tf.get_variable(
                "lsf", shape=[], initializer=tf.zeros_initializer, dtype=self.dtype
            )

        for i in range(n_nodes):
            with tf.variable_scope(f"Node_{i}"):
                gp_node = GPNode(
                    node_id=i,
                    W=W,
                    n_inducing_points=self.n_inducing_points,
                    n_points=n_points,
                    input_d=self.input_d,
                    first_gp_layer=first_gp_layer,
                    jitter=jitter,
                    shared_layer_tf_variables=self.shared_layer_tf_variables,
                    q_initializations=q_initializations,
                    z_initializations=self.z_initializations,
                    seed=seed,
                    dtype=self.dtype,
                )
                self.add_node(gp_node)

    def stack_on_previous_layer(self, previous_layer):
        BaseLayer.stack_on_previous_layer(self, previous_layer)

    def initialize_params_layer(self):
        """Returns the operations to initialize all params

        """
        assert self.stacked

        current_nodes = self.get_node_list()
        tf_operations = []
        tf_operations += self.build_initialize_z_layer()
        tf_operations += self.build_initialize_kernel_params()
        for _, current_node in enumerate(current_nodes):
            with tf.control_dependencies(tf_operations):
                tf_operations += current_node.initialize()
        self.initialized = True
        return tf_operations

    def get_layer_contribution_to_energy(self):
        # assert self.initialized == True
        current_nodes = self.get_node_list()
        contribution = 0.0
        for node in current_nodes:
            contribution += node.get_node_contribution_to_energy()
        return contribution

    def build_initialize_z_layer(self):
        if self.shared_layer_tf_variables.get("Z") is None:
            # If we are not sharing we don't have anything to initialize here
            return []
        if self.z_initializations is None:
            # dummy: Initialize in interval -1, 1
            # TODO: The ideal thing would be to detect uninitialized z in gp_network.py
            # and initialize them there with Kmeans.
            self.z_initializations = tf.tile(
                tf.lin_space(-1.0, 1.0, self.n_inducing_points)[:, None],
                (1, self.input_d),
            )
        return [self.shared_layer_tf_variables["Z"].assign(self.z_initializations)]

    def build_initialize_kernel_params(self):
        init_ops = []
        if self.share_kernel_params:
            lls = tf.zeros(
                shape=[int(self.shared_layer_tf_variables["lls"].shape[0])],
                dtype=self.dtype,
            )
            lsf = tf.zeros(1, dtype=self.dtype)[0]
            init_ops += [self.shared_layer_tf_variables["lls"].assign(lls)]
            init_ops += [self.shared_layer_tf_variables["lsf"].assign(lsf)]
        return init_ops

    def get_params(self):
        params = []
        for node in self.get_node_list():
            params = params + node.get_params()
        # Add remaining parameters shared within the layer
        # (or don't add anything if nothing is shared)
        params += list(self.shared_layer_tf_variables.values())
        return params

