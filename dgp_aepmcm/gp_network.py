##
# This class represents a node within the network
#
import sys
import time
import warnings
from collections import deque

import numpy as np
import tensorflow as tf

from dgp_aepmcm.layers.gp_layer import GPLayer
from dgp_aepmcm.layers.input_layer import InputLayer
from dgp_aepmcm.layers.noise_layer import NoiseLayer
from dgp_aepmcm.layers.output_layer_classification import OutputLayerClassification
from dgp_aepmcm.layers.output_layer_regression import OutputLayerRegression

from .utils import (
    ProblemType,
    calculate_ETA_str,
    extend_dimension_if_1d,
    memory_used,
    valid_q_initializations,
)


class DGPNetwork:
    """Creates a new Deep GP network using Approximate Expectation propagation and Monte Carlo Methods

    Args:
        x_train (ndarray): Training points (X)
        y_train (ndarray): Training targets (y)
        inducing_points (ndarray): If not None, initializations for the inducing points (Z) of the GP nodes
        share_z_within_layer (Boolean): If True all the nodes in the GP same layer share
            the same inducing points
        share_kernel_params_within_layer (Boolean): If True all the nodes in the same GP layer
            share the same kernel parameters but still using ARD kernel.
        n_samples_training (int): Number of samples to use when training
        n_samples_prediction (int): Number of samples to use when predicting
        show_debug_info (Boolean): Show Epoch information when training
        sacred_exp (): _run variable of sacred experiment information,
            see: http://sacred.readthedocs.io/en/latest/collected_information.html
        seed (int): Seed to use in random number generation functions
        jitter (float): Jitter level to add to the diagonal of Kxx, bigger jitters improve numerical stability
        minibatch_size (int): Minibatch size to use when initializing, training and predicting.
            Smaller minibatches makes the training use less memory.
        dtype (type): Type to use for inputs (X) of the network. Either np.float32/np.float64.
            float64 will make the network more stable but slower.
    """

    def __init__(
        self,
        x_train,
        y_train,
        inducing_points=None,
        share_z_within_layer=False,
        share_kernel_params_within_layer=False,
        n_samples_training=20,
        n_samples_prediction=100,
        show_debug_info=True,
        sacred_exp=None,
        seed=None,
        jitter=1e-5,
        minibatch_size=100,
        dtype=np.float32,
    ):
        # Sometimes the Tensorflow graph is not deleted when the class is destroyed.
        tf.reset_default_graph()
        self.seed = seed
        self.dtype = dtype
        if seed is not None:
            print(f"Random seed set: {seed}")
            tf.set_random_seed(seed)
            np.random.seed(seed)

        self.x_train = x_train
        self.y_train = y_train
        self.inducing_points = inducing_points
        self.share_z = share_z_within_layer
        self.share_kernel_params = share_kernel_params_within_layer
        self.show_debug_info = show_debug_info
        # To store sacred experiments data (_run dictionary).
        # More info: https://sacred.readthedocs.io/en/latest/collected_information.html
        self.sacred_exp = sacred_exp

        self.x_train = extend_dimension_if_1d(self.x_train)
        self.y_train = extend_dimension_if_1d(self.y_train)

        self.n_points = self.x_train.shape[0]
        self.problem_dim = self.x_train.shape[1]
        # Minibatch size to use in the network and reduce memory usage.
        self.minibatch_size = min(self.n_points, minibatch_size)

        # Three possible values, regression, bin_classification, multi_classification
        self.problem_type = None
        self.jitter = jitter

        if self.inducing_points is not None:
            self.inducing_points = extend_dimension_if_1d(self.inducing_points)
            assert (
                self.inducing_points.shape[1] == self.x_train.shape[1]
            ), "The inducing points dimensions must be the same as the X dimensions"
            self.inducing_points = self.inducing_points.astype(self.dtype)
            self.z_running_tf = self.inducing_points

        self.x_tf = tf.placeholder(
            self.dtype, name="x_input", shape=[None, self.x_train.shape[1]]
        )

        # If targets are integer -> classification problem
        #   If targets are -1 and 1 -> binary classification
        #   If targets have values from 0, 1, 2,.. n_classes - 1 -> multiclass classification
        if np.sum(np.mod(self.y_train, 1)) == 0:
            # There is no decimal in y training , we are probably in a classification problem
            self.y_train = self.y_train.astype(np.int32)
        if np.issubdtype(self.y_train.dtype, np.integer):
            self.n_classes = np.max(self.y_train) + 1
            # self.n_classes = len(np.unique(self.y_train)) # This one works even if the classes start at 1
            y_type = tf.int32
            if self.show_debug_info:
                print(
                    f"Creating DGP network for classification problem with {self.n_classes} classes"
                )
            if self.n_classes == 2:
                self.problem_type = ProblemType.BINARY_CLASSIFICATION
            else:
                self.problem_type = ProblemType.MULTICLASS_CLASSIFICATION
        else:
            if self.show_debug_info:
                print(f"Creating DGP network for regression problem")
            self.problem_type = ProblemType.REGRESSION
            y_type = self.dtype

        # TODO: merge this two placeholders into one. As in x_tf
        self.y_train_tf = tf.placeholder(
            y_type, name="y_training", shape=[None, self.y_train.shape[1]]
        )
        self.y_test_tf = tf.placeholder(
            y_type, name="y_training", shape=[None, self.y_train.shape[1]]
        )

        self.y_train_mean_tf = None
        self.y_train_std_tf = None

        self.layers = []
        self.initialized = False
        self._predict_function = None
        self.session_saved = False

        self.n_samples_dict = {
            "training": n_samples_training,  # num samples for training
            "prediction": n_samples_prediction,  # num samples for prediction
        }

        # Placeholder for the status of the network.
        # 1 -> Training 0 -> Prediction
        # Tells the network the right number of samples to use (training or prediction)
        # and uses either the cavity (training) or the posterior (prediction) in the GP node
        self.network_set_for_training_tf = tf.placeholder(
            self.dtype, shape=(), name="network_set_for_training"
        )

        self.x_running_tf = tf.cast(self.x_train, self.dtype)
        self.sess = tf.Session()
        self.saver = None
        self.objective_energy_function = None
        self.trainable_params = None
        self.gradient_optimization_step = None

    def add_input_layer(self):
        """Adds an input layer to the network.

        The input layer is in charge of replicating the x_train of shape (N,D) to shape (S,N,D)

        """
        assert not self.layers, "Network should be empty"
        with tf.variable_scope("Input_layer"):
            new_layer = InputLayer(
                self.x_tf,
                self.problem_dim,
                self.n_samples_dict,
                self.network_set_for_training_tf,
            )
            self._stack_new_layer(new_layer)

    def add_noise_layer(self, noise_initial_value=0.01):
        """Adds noise to the variance of the output of the layer

        Args:
            noise_initial_value (float): Initial value for the noise
        """
        assert self.layers, "Network should have an input node"
        # TODO: Reduce default noise?
        new_layer = NoiseLayer(self.dtype, noise_initial_value)
        self._stack_new_layer(new_layer, self.layers[-1])

    def add_gp_layer(
        self, n_inducing_points, n_nodes=1, q_initializations="random", W=None
    ):
        """Adds a Gaussian processes layer

        Args:
            n_inducing_points (int): Number of inducing points (Z)
            n_nodes (int): Number of GP nodes of the layer, the number of nodes will be the output dim. of the layer
            q_initializations (str): Initializations of the posterior approximation q(u) params. Valid values are:
                'random' (default): Mean and covariance initialized from random normal.
                'deterministic': Mean initialized to mean of the prior p(u) and cov. to 1e-5 * Kzz (1e-5 * prior cov)
                'prior': Mean and cov. initialized to the prior covariance.
            W (ndarray): Mean function weights of the GP m(x) = XW, if None, the identity matrix will be used
        """
        if q_initializations not in valid_q_initializations():
            raise ValueError(
                f"initializations should take a value from {valid_q_initializations()}"
            )
        assert self.layers, "Network should have an input node"

        with tf.variable_scope(f"Layer_{len(self.layers)}_GP"):
            is_first_layer = len(self.layers) == 1
            # The dim of the layer is the number of nodes in the last one
            input_dim_layer = self.layers[-1].n_nodes
            output_dim_layer = n_nodes
            Z = None
            if self.inducing_points is not None:
                Z = tf.identity(self.z_running_tf)
            # set mean function weights of the layer.
            # W should have the same dimension [1] as the number of nodes in the layer
            if W is None:
                W = self._linear_mean_function(input_dim_layer, output_dim_layer)
            else:
                W = tf.cast(W, self.dtype)
                self.x_running_tf = tf.matmul(self.x_running_tf, W)
                if self.inducing_points is not None:
                    self.z_running_tf = self.z_running_tf @ W

            assert W.shape == [input_dim_layer, output_dim_layer], (
                f"The given mean weights must be of shape [input_d({input_dim_layer}), output_d({output_dim_layer})], "
                f"Given: {W.shape}"
            )
            new_layer = GPLayer(
                W=W,
                n_inducing_points=n_inducing_points,
                n_points=self.n_points,
                n_nodes=output_dim_layer,
                input_d=input_dim_layer,
                first_gp_layer=is_first_layer,
                jitter=self.jitter,
                share_z=self.share_z,
                share_kernel_params=self.share_kernel_params,
                q_initializations=q_initializations,
                z_initializations=Z,
                seed=self.seed,
                dtype=self.dtype,
            )
            self._stack_new_layer(new_layer, self.layers[-1])

    def _linear_mean_function(self, input_dim_layer, output_dim_layer):
        """ Sets the W for the mean function m(X) = XW

        The last GP layer will have m(X) = 0. This method is based on:
        Doubly Stochastic Variational Inference for Deep Gaussian Processes https://arxiv.org/abs/1705.08933

        Args:
            input_dim_layer (int): Input dimension to the layer. (Number of nodes in the last layer)
            output_dim_layer (int): Dimension of the layer. (Number of nodes in the layer)
        """
        if input_dim_layer == output_dim_layer:
            W = tf.eye(output_dim_layer, dtype=self.dtype)
        elif output_dim_layer > input_dim_layer:
            zeros = tf.zeros(
                (input_dim_layer, output_dim_layer - input_dim_layer), dtype=self.dtype
            )
            W = tf.concat([tf.eye(input_dim_layer, dtype=self.dtype), zeros], 1)
            self.x_running_tf = tf.matmul(self.x_running_tf, W)
            if self.inducing_points is not None:
                self.z_running_tf = self.z_running_tf @ W
        elif output_dim_layer < input_dim_layer:
            _, _, V = tf.svd(self.x_running_tf)
            # Using the first output_dim_layer values of the input X
            W = tf.transpose(V[:output_dim_layer, :])
            self.x_running_tf = tf.matmul(self.x_running_tf, W)
            if self.inducing_points is not None:
                self.z_running_tf = self.z_running_tf @ W
        return W

    def _set_mean_function_last_layer(self):
        # Set the mean funcion of the last GP layer to Zero
        for layer in reversed(self.layers):
            if isinstance(layer, GPLayer):
                for node in layer.get_node_list():
                    node.W = tf.zeros_like(node.W)
                return

    def _stack_new_layer(self, new_layer, previous_layer=None):
        # Previous layer should be None only when adding the input layer
        if previous_layer is not None:
            new_layer.stack_on_previous_layer(previous_layer)
        self.layers.append(new_layer)

    def add_output_layer_regression(self):
        """ Add an output layer for regression to the network

        This mean that a Gaussian Likelihood is used.

        """
        assert self.layers, "Network should have an input node"
        self._require_normalized_y()
        with tf.variable_scope(f"Layer_{len(self.layers)}_Out"):
            new_layer = OutputLayerRegression(
                self.y_train_tf,
                self.y_test_tf,
                self.y_train_mean_tf,
                self.y_train_std_tf,
                self.n_samples_dict,
                self.dtype,
            )
            new_layer.stack_on_previous_layer(self.layers[-1])
            self.layers.append(new_layer)

    def add_output_layer_classification(
        self, *, use_norm_cdf=False, noise_in_labels=False, noise_in_labels_trainable=True
    ):
        """ Add an output layer for regression to the network

        The likelihood is given by a step function, that combined with a Gaussian dist.
        for the output of the layers, yields a Gaussian cdf.

        Args:
            use_norm_cdf (Boolean): Add bias term (+1) to the variance of f^L (+0 if False).
                if use_norm_cdf == True then likelihood p(y | f^L) will be norm.cdf(y_train * f^L)
                if use_norm_cdf == False then likelihood p(y | f^L) will be heavyside(y_train * f^L)
                only used in binary classification.
            noise_in_labels (Boolean): If true the likelihood will take into account
                that there may be wrong labeled examples. Using a robust multiclass likelihood (as in GPflow)
            noise_in_labels_trainable (Boolean): Specifies if the noise in labels is a trainable parameter.
                For fair comparison with DGP-VI it should be set to False,
                for other tasks it should be set to True as it makes the network more robust

        """
        assert self.layers, "Network should have an input node"
        variance_bias = (
            tf.constant(1.0, dtype=self.dtype)
            if use_norm_cdf
            else tf.constant(0.0, dtype=self.dtype)
        )
        with tf.variable_scope("Layer_{}_Out".format(len(self.layers))):
            new_layer = OutputLayerClassification(
                self.y_train_tf,
                self.y_test_tf,
                self.n_samples_dict,
                self.n_classes,
                variance_bias,
                noise_in_labels,
                noise_in_labels_trainable,
                self.dtype,
            )
            new_layer.stack_on_previous_layer(self.layers[-1])
            self.layers.append(new_layer)

    def add_output_layer_regression_multioutput(self, n_outputs):
        raise NotImplementedError()
        # assert self.layers, "Network should have an input node"
        # new_layer = OutputLayerRegressionMultioutput(self.y_train, n_outputs)
        # new_layer.stack_on_previous_layer(self.layers[-1])
        # self.layers.append(new_layer)

    def _require_normalized_y(self):
        # This function should be called when the network requires normalized observations
        # (regression)
        if self.y_train_mean_tf is None:
            self.y_train_mean_tf = tf.placeholder(
                self.dtype, name="y_train_mean", shape=(1,)
            )
        if self.y_train_std_tf is None:
            self.y_train_std_tf = tf.placeholder(
                self.dtype, name="y_train_std", shape=(1,)
            )

    def _initialize_network(self, learning_rate=1e-3):
        assert len(self.layers) > 1
        if self.initialized:
            return
        if self.show_debug_info:
            print("Initializing network")

        self._set_mean_function_last_layer()

        # Do a forward pass trough the network to 'connect the graph'
        self.objective_energy_function = -self._get_network_energy()

        # Params to optimize
        self.trainable_params = self.get_params()
        self.gradient_optimization_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(self.objective_energy_function, var_list=self.trainable_params)

        self.sess.run(tf.global_variables_initializer())
        # All inits operations remaining
        tf_operations = []
        ops_returned = None
        for layer in self.layers:
            with tf.control_dependencies(tf_operations):
                ops_returned = layer.initialize_params_layer()
                if ops_returned is not None:
                    tf_operations += ops_returned

        # If minibatch size is smaller than N
        # Use part of the data to initialize the network and be memory efficient
        batch_indexes = np.random.choice(
            self.n_points, min(int(self.minibatch_size), self.n_points), replace=False
        )
        self.sess.run(
            tf_operations,
            feed_dict={
                self.x_tf: self.x_train[batch_indexes],
                self.y_train_tf: self.y_train[batch_indexes],
                self.network_set_for_training_tf: 1.0,
            },
        )
        self._load_functions_to_graph()
        self.initialized = True

    def _load_functions_to_graph(self):
        """Load Symbolic tensorflow functions
        """
        # Predict function
        self._predict_function = self.layers[-1].get_predicted_values()
        if self.problem_type == ProblemType.REGRESSION:
            # TODO: Implement some of these for classification
            # Calculate rmse function
            self._rmse_likelihood_function = self.layers[-1].calculate_loglikehood_rmse()
            # Sample from predictive dist.
            self._sample_from_predictive_function = self.layers[
                -1
            ].sample_from_predictive_distribution()
            # Get PDF for point function
            self.y_range_tf = tf.placeholder(
                self.dtype, name="y_range", shape=[None, self.y_train.shape[1]]
            )
            self._pdf_function = (
                self.layers[-1].get_predictive_distribution_fixed_x(self.y_range_tf),
            )
        if self.problem_type == ProblemType.BINARY_CLASSIFICATION:
            self._log_likelihood_function = self.layers[-1].calculate_log_likelihood()
            self._sample_from_last_layer = self.layers[-1].sample_from_latent()

        if self.problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
            self._log_likelihood_function = self.layers[-1].calculate_log_likelihood()
        self._init_saver()

    def _get_network_energy(self):
        """Returns the tensorflow operation to calculate the energy of the network
        The energy is the approximation to the marginal likelihood of the AEP algorithm

        Returns:
            Tensor -- Symbolic operation to calculate the energy
        """
        energy = 0.0
        for layer in self.layers:
            layer.forward_pass_computations()
            energy += layer.get_layer_contribution_to_energy()
        return energy[0, 0]

    def get_params(self):
        """Returns all trainable parameters of the network

        Returns:
            list -- List of Tensor, with all the parameters
        """
        assert len(self.layers) > 1
        if self.trainable_params is not None:
            return self.trainable_params
        params = []
        for layer in self.layers:
            params += layer.get_params()

        return params

    def train_via_adam(self, max_epochs=1000, learning_rate=1e-3, step_callback=None):
        """ Finalizes the graph and trains the DGP AEPMCM network using Adam optimizer.

        Args:
            max_epochs (int): Maximun number of epochs to train for.
                An epoch is a full pass through all the minibatches (whole dataset)
            learning_rate (float): Learning rate to use. Default = 1e-3
            step_callback (function): If set, function to call every gradient step.
                This function should accept at least one parameter, the iteration number.
        """
        assert len(self.layers) > 1

        if self.show_debug_info:
            print("Compiling adam updates")

        self._initialize_network(learning_rate)

        # self.sess.graph.finalize()

        # Main loop of the optimization
        n_batches = int(np.ceil(self.n_points / self.minibatch_size))
        if self.show_debug_info:
            print(
                f"Training for {max_epochs} epochs, {max_epochs * n_batches} iterations"
            )
        sys.stdout.flush()

        start = time.time()
        # Object that keeps maxlen epoch times, for ETA prediction.
        last_epoch_times = deque(maxlen=20)
        for j in range(max_epochs):
            shuffle = np.random.choice(self.n_points, self.n_points, replace=False)
            shuffled_x_train = self.x_train[shuffle, :]
            shuffled_y_train = self.y_train[shuffle, :]
            avg_energy = 0.0
            start_epoch = time.time()
            for i in range(n_batches):
                start_index = i * self.minibatch_size
                end_index = min((i + 1) * self.minibatch_size, self.n_points)
                minibatch_x = shuffled_x_train[start_index:end_index, :]
                minibatch_y = shuffled_y_train[start_index:end_index, :]
                current_energy = self.sess.run(
                    [self.gradient_optimization_step, self.objective_energy_function],
                    feed_dict={
                        self.x_tf: minibatch_x,
                        self.y_train_tf: minibatch_y,
                        self.network_set_for_training_tf: 1.0,
                    },
                )[1]
                if step_callback is not None:
                    step_callback(self, j * n_batches + i)
                avg_energy += current_energy / (minibatch_x.shape[0] * n_batches)
            elapsed_time_epoch = time.time() - start_epoch
            last_epoch_times.append(elapsed_time_epoch)

            if self.show_debug_info:
                eta = calculate_ETA_str(last_epoch_times, j, max_epochs)
                print(
                    "Epoch: {: <4}| Energy: {: <11.6f} | Time: {: >8.4f}s | Memory: {: >2.2f} GB | ETA: {}".format(
                        j, avg_energy, elapsed_time_epoch, memory_used(), eta
                    )
                )
                sys.stdout.flush()
            if self.sacred_exp is not None:
                self.sacred_exp.log_scalar("train.energy", round(avg_energy, 4))
        elapsed_time = time.time() - start

        if self.show_debug_info:
            print("Total time: {}".format(elapsed_time))

        # Log final energy to sacred
        if self.sacred_exp is not None:
            if self.sacred_exp.info.get("last_train_energies") is None:
                self.sacred_exp.info.update(
                    {"last_train_energies": [round(avg_energy, 4)]}
                )
            else:
                self.sacred_exp.info.get("last_train_energies").append(
                    round(avg_energy, 4)
                )

    def predict(self, x_test):
        """ Returns predictions for a given x

        Args:
            x_test (ndarray): K x D matrix with locations for predictions.
                With K the number of test points and D the dimension.
                D should be the same as the one in the original training data.
        """
        x_test = extend_dimension_if_1d(x_test)
        assert x_test.shape[1] == self.problem_dim

        x_test = x_test.astype(self.dtype)
        # Use minibatches to predic
        n_batches = int(np.ceil(x_test.shape[0] / self.minibatch_size))
        pred, uncert = [], []
        current_batch = 0
        for x_test_batch in np.array_split(x_test, n_batches):
            if self.show_debug_info and n_batches > 1:
                current_batch += 1
                print(f"Predicting batch {current_batch}/{n_batches}")
            pred_batch, uncert_batch = self.sess.run(
                self._predict_function,
                feed_dict={
                    self.x_tf: x_test_batch,
                    self.network_set_for_training_tf: 0.0,
                },
            )
            pred.append(pred_batch)
            uncert.append(uncert_batch)
        pred_uncert_values = np.concatenate(pred, 0), np.concatenate(uncert, 0)

        return pred_uncert_values

    def sample_from_predictive_distribution(self, x_locations):
        assert x_locations.shape[1] == self.problem_dim
        x_locations = x_locations.astype(self.dtype)
        samples = self.sess.run(
            self._sample_from_predictive_function,
            feed_dict={self.x_tf: x_locations, self.network_set_for_training_tf: 0.0},
        )
        return samples

    def get_predictive_distribution_for_x(self, x_value, y_range):
        """ Returns the probability of each y value for a fixed x. p(y | x)

        It returns the predictive distribution for a fixed x.
        Useful to plot the PDF of the predictive distribution

        Args:
            x_value (ndarray): Single point to which calculate the PDF
            y_range (ndarray): All the plausible y values to test. suggested: np.linspace()

        """
        assert x_value.shape[1] == self.problem_dim

        x_value = x_value.astype(self.dtype)
        pdf = self.sess.run(
            self._pdf_function,
            feed_dict={
                self.x_tf: x_value,
                self.y_range_tf: y_range,
                self.network_set_for_training_tf: 0.0,
            },
        )
        return pdf[0]

    def calculate_log_likelihood(
        self, x_test, y_test, y_train_mean=None, y_train_std=None
    ):
        if self.problem_type == ProblemType.REGRESSION:
            raise NotImplementedError()
        elif (
            self.problem_type == ProblemType.BINARY_CLASSIFICATION
            or self.problem_type == ProblemType.MULTICLASS_CLASSIFICATION
        ):
            n_batches = int(np.ceil(x_test.shape[0] / self.minibatch_size))
            lik = []
            for X_batch, Y_batch in zip(
                np.array_split(x_test, n_batches), np.array_split(y_test, n_batches)
            ):
                l = self.sess.run(
                    self._log_likelihood_function,
                    feed_dict={
                        self.x_tf: X_batch,
                        self.y_test_tf: Y_batch,
                        self.network_set_for_training_tf: 0.0,
                    },
                )
                lik.append(l)
            # (N, 1), still need to calculate the average likelihood for all the dataset
            lik = np.concatenate(lik, 0)
            return np.mean(lik)
        else:
            raise NotImplementedError()

    def save_model(self, path_to_save, name):
        save_path = self.saver.save(self.sess, f"{path_to_save}/{name}.ckpt")
        print(f"Model saved in path: {save_path}")

    def restore_model(self, model_path, name):
        if not self.initialized:
            self._initialize_network()
        self.saver.restore(self.sess, f"{model_path}/{name}.ckpt")

    def _init_saver(self):
        if self.saver is None:
            self.saver = tf.train.Saver()

    def calculate_loglikehood_rmse(self, x_test, y_test, y_train_mean, y_train_std):
        # TODO: As we will normally want log likelihood for classification too
        # this function should be separated.
        # The calculate_log_likelihood valid for all kind of problems
        # and the RMSE one valid just for regression.

        # We expect unnormalized y_test
        if not np.allclose(np.mean(x_test), 0, atol=0.1) or not np.allclose(
            np.std(x_test), 1.0, atol=0.1
        ):
            warnings.warn(
                f"x_test should be normalized current mean = {np.mean(x_test)} and std = {np.std(x_test)}"
            )

        if self.problem_type != ProblemType.REGRESSION:
            raise NotImplementedError()
        n_batches = int(np.ceil(x_test.shape[0] / self.minibatch_size))
        lik, sq_diff = [], []
        for X_batch, Y_batch in zip(
            np.array_split(x_test, n_batches), np.array_split(y_test, n_batches)
        ):
            l, sq = self.sess.run(
                self._rmse_likelihood_function,
                feed_dict={
                    self.x_tf: X_batch,
                    self.y_test_tf: Y_batch,
                    self.y_train_mean_tf: y_train_mean.flatten(),
                    self.y_train_std_tf: y_train_std.flatten(),
                    self.network_set_for_training_tf: 0.0,
                },
            )
            lik.append(l)
            sq_diff.append(sq)
        # (N, 1), still need to calculate the average likelihood for all the dataset
        lik = np.concatenate(lik, 0)
        sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=self.dtype)
        return np.average(lik), np.average(sq_diff) ** 0.5

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.sess.close()
        tf.reset_default_graph()
