import numpy as np
import tensorflow as tf

from dgp_aepmcm.kernel.gauss import compute_kernel
from dgp_aepmcm.nodes.base_node import BaseNode
from dgp_aepmcm.utils import casting, valid_q_initializations


##
# This class represents a GP node in the network
#


class GPNode(BaseNode):
    def __init__(
        self,
        node_id,
        W,
        n_inducing_points,
        n_points,
        input_d,
        first_gp_layer,
        jitter,
        shared_layer_tf_variables,
        q_initializations,
        z_initializations,
        seed=None,
        dtype=None,
    ):
        """Instantiates a GP node

        Args:
            node_id (int): Id of the node inside the layer
            W (Tensor): Matrix to multiply the inputs to use ass mean function of the GP. m(x)= XW
            n_inducing_points (int): Number of inducing points to use on the node
            n_points (int): Total number of training points (used for cavity computation)
            input_d (int): Input dimensions, either dimensions of the problem if first layer
                or number of nodes on the last layer
            first_gp_layer (Boolean): if this GP is on the first layer of the network.
                Used for initialization
            jitter (float): Jitter level to add to the diagonal of Kxx, bigger jitters improve numerical stability
            shared_layer_tf_variables (dict): All the variables shared withing the layer.
                Possible ones: Z, lls, lsf.
                It can be empty meaning all nodes have their own variables (no variable shared).
            q_initializations (str): Initializations of the posterior approximation q(u) params. Valid values are:
                'random' (default): Mean and covariance initialized from random normal.
                'deterministic': Mean initialized to mean of the prior p(u) and cov. to 1e-5 * Kzz (1e-5 * prior cov)
                'prior': Mean and cov. initialized to the prior covariance.
            z_initializations: (ndarray) Values to initialize Z. Only used if Z is not shared within layer
            seed (int): seed to use in random functions.
            dtype (type): Type to use in tf operations.
        """
        # Note: n_points are the total number of training points (that is used for cavity computation)

        BaseNode.__init__(self)
        self.id = node_id
        self.W = W
        self.n_inducing_points = n_inducing_points
        self.n_points = n_points
        self.input_d = input_d
        self.first_gp_layer = first_gp_layer
        # Level of jitter to use  (added to the diagonal of Kxx)
        self.jitter = jitter
        self.shared_layer_tf_variables = shared_layer_tf_variables
        self.q_initializations = q_initializations
        self.z_initializations = z_initializations
        self.seed = seed
        if dtype is None:
            raise ValueError("dtype should be specified (gp_layer)")
        self.dtype = dtype
        # These are the actual parameters of the posterior distribution being optimzied
        # covCavity = (Kzz^-1 + LParamPost LParamPost^T * (n - 1) / n)
        # and meanCavity = covCavity mParamPost * (n - 1) / n
        self.LParamPost = tf.get_variable(
            "LParamPost",
            shape=[n_inducing_points, n_inducing_points],
            initializer=tf.zeros_initializer,
            dtype=self.dtype,
        )
        self.mParamPost = tf.get_variable(
            "mParamPost",
            shape=[n_inducing_points, 1],
            initializer=tf.zeros_initializer,
            dtype=self.dtype,
        )
        # Log lengthscale (rbf kernel)
        self.lls = self.shared_layer_tf_variables.get(
            "lls",
            tf.get_variable(
                "lls",
                shape=[self.input_d],
                initializer=tf.zeros_initializer,
                dtype=self.dtype,
            ),
        )
        self.lsf = self.shared_layer_tf_variables.get(
            "lsf",
            tf.get_variable(
                "lsf", shape=[], initializer=tf.zeros_initializer, dtype=self.dtype
            ),
        )

        self.Z = self.shared_layer_tf_variables.get(
            "Z",
            tf.get_variable(
                "Z",
                shape=[self.n_inducing_points, input_d],
                initializer=tf.zeros_initializer,
                dtype=self.dtype,
            ),
        )
        self.set_for_training = np.array(1.0)

    @property
    def W(self):
        return self.__W

    @W.setter
    def W(self, W):
        self.__W = W

    def initialize(self):
        input_means, input_vars = self.find_input()  # both size (S,N,D)
        initialization_list = []
        ##########################################
        # Inducing points initializations
        ##########################################
        if self.shared_layer_tf_variables.get("Z") is None:
            # If we are not in the first layer, we initialize in the -1, 1 range the inducing points
            if self.z_initializations is None:
                if self.first_gp_layer:
                    # First Layer
                    self.z_initializations = tf.random_shuffle(
                        input_means, seed=self.seed
                    )[0, : self.n_inducing_points, :]
                else:
                    # Not first layer and we don't have any initialization to use
                    self.z_initializations = tf.tile(
                        tf.reshape(
                            tf.lin_space(-1.0, 1.0, self.n_inducing_points),
                            shape=[self.n_inducing_points, 1],
                        ),
                        [1, self.input_d],
                    )
            initialization_list.append(self.Z.assign(self.z_initializations))

        ##########################################
        # Kernel params initializations
        ##########################################
        if self.shared_layer_tf_variables.get("lls") is None:
            # This is the actual initialization for DGP-VI
            lls = tf.zeros([self.input_d], dtype=self.dtype)
            if self.first_gp_layer:
                # We initialize the lls on the first layer to the log of the median of the squared distance
                M = tf.reduce_sum(input_means[0] ** 2, 1)[:, None] * tf.ones(
                    [1, tf.shape(input_means)[1]], dtype=self.dtype
                )
                dist = (
                    M
                    - 2 * tf.matmul(input_means[0], input_means[0], transpose_b=True)
                    + tf.transpose(M)
                )
                lls = tf.log(self.calculate_median(dist)) * tf.ones(
                    [tf.shape(input_means)[2]], dtype=self.dtype
                )
            initialization_list.append(self.lls.assign(lls))
        if self.shared_layer_tf_variables.get("lsf") is None:
            initialization_list.append(self.lsf.assign(tf.zeros(1, dtype=self.dtype)[0]))
        ##########################################
        # Initialize EP posterior params
        ##########################################
        if self.q_initializations == 'random':
            LParamPost_init = tf.random_normal(
                shape=[self.n_inducing_points, self.n_inducing_points],
                seed=self.seed,
                dtype=self.dtype,
            )
            mParamPost_init = tf.random_normal(
                shape=[self.n_inducing_points, 1], seed=self.seed, dtype=self.dtype
            )
            initialization_list.append(self.LParamPost.assign(LParamPost_init))
            initialization_list.append(self.mParamPost.assign(mParamPost_init))
        elif self.q_initializations == 'prior' or self.q_initializations == 'deterministic':
            ##########################################
            # Initialize EP posterior params with those of the prior p(u)
            ##########################################
            # We want to initialize the mean and cov of the posterior q(u)
            # to the mean and cov of the prior p(u) = N(m(z), Kzz).

            # For the inner layers (all layers except the last gp layer)
            # we will initialize the post. cov. to Kzz * 1e-5.
            # For the last layer we initialize the cov to Kzz
            Kzz = (
               compute_kernel(self.lls, self.lsf, self.Z, dtype=self.dtype)
               + tf.eye(self.n_inducing_points, dtype=self.dtype) * self.jitter
            )
            KzzInv = tf.cholesky_solve(
               tf.cholesky(Kzz, name="kzzChol"),
               tf.eye(self.n_inducing_points, dtype=self.dtype),
               name="KzzInv",
            )
            # inv_scaling is the inverse of the cte multiplying Kzz
            if self.q_initializations == 'prior':
                inv_scaling = 1e-5
                LParamPost_init = (
                    tf.eye(self.n_inducing_points, dtype=self.dtype) * inv_scaling
                )
            if self.q_initializations == 'deterministic':
                inv_scaling = 1e5
                # As LParamPost is an upper triangular matrix we have to tranpose the cholesky factor
                LParamPost_init = tf.transpose(
                    tf.cholesky(
                        (
                            tf.eye(self.n_inducing_points, dtype=self.dtype) * inv_scaling
                            - tf.eye(self.n_inducing_points, dtype=self.dtype)
                        )
                        @ KzzInv
                    )
                )
            with tf.control_dependencies(initialization_list):
               mean_function_z = tf.matmul(self.Z, self.W)[:, self.id : self.id + 1]
               mParamPost_init = (
                   (inv_scaling - 1)
                   * tf.eye(self.n_inducing_points, dtype=self.dtype)
                   @ KzzInv
                   @ mean_function_z
               )
               # Because the diagonal of LParamPost_init will be subtituted by e**diag(LParamPost_init)
               # we subtitute it by the log
               LParamPost_init = (
                   LParamPost_init
                   - tf.diag(tf.diag_part(LParamPost_init))
                   + tf.diag(tf.log(tf.diag_part(LParamPost_init)))
               )

               initialization_list.append(self.LParamPost.assign(LParamPost_init))
               initialization_list.append(self.mParamPost.assign(mParamPost_init))

        with tf.control_dependencies(initialization_list):
            self.calculate_output(input_means, input_vars)

        return initialization_list

    def calculate_output(self, input_means, input_vars):
        # assert self.n_samples_to_propagate is not None, "Network must be set for training or prediction first"
        # Both inputs to this function are of size (S,N,D)
        # with D being either the number of dimensions of the problem (for first layer)
        # or the number of nodes in the last layer
        self.input_means = input_means
        self.input_vars = input_vars
        # We sample from a normal dist. Independent in S,N,D
        self.input_samples = tf.random_normal(
            tf.shape(self.input_means),
            mean=self.input_means,
            stddev=tf.sqrt(self.input_vars),
            seed=self.seed,
            dtype=self.dtype,
        )

        S, N, D, M = (
            tf.shape(self.input_samples)[0],
            tf.shape(self.input_samples)[1],
            self.input_d,
            self.n_inducing_points,
        )

        # Compute the kernels matrix
        input_samples_flat = tf.reshape(self.input_samples, [S * N, D])
        # The kernel returns shape (S*N, M) and we convert it to the correct (S,N,M)
        # import pdb; pdb.set_trace()
        self.Kxz = tf.reshape(
            compute_kernel(
                self.lls, self.lsf, input_samples_flat, self.Z, dtype=self.dtype
            ),
            [S, N, M],
        )

        # The kernel returns (M,M) and we want (S,M,M)
        # Tensorflow doesn't have tile minimization implemented
        # self.Kzz = tf.tile(compute_kernel(self.lls, self.lsf, self.Z)[None,:,:], [S, 1, 1])
        self.Kzz = (
            compute_kernel(self.lls, self.lsf, self.Z, dtype=self.dtype)
            + tf.eye(M, dtype=self.dtype) * self.jitter
        )

        self.mean_function_z = tf.matmul(self.Z, self.W)[:, self.id : self.id + 1]
        self.mean_function_x = tf.matmul(
            self.input_samples, tf.tile(self.W[None, :, :], [S, 1, 1])
        )[:, :, self.id : self.id + 1]

        # (M,M)
        # TODO: Sometimes it fails here for some datasets, adding jitter solves the problem but does not seems
        # like the right solution
        self.KzzInv = tf.cholesky_solve(
            tf.cholesky(self.Kzz, name="kzzChol"),
            tf.eye(M, dtype=self.dtype),
            name="KzzInv",
        )
        # self.KzzInv = tf.matrix_inverse(self.Kzz, name="KzzInv")

        # LParamPost_tri is an upper traingular matrix
        # with exponential of LParamPost in the diagonal
        # and LParamPost values in the upper part outside the diagonal
        # LParamPost_tri <- Upper diagonal part - Diagonal + exponential of diagonal
        # (M,M)
        LParamPost_tri = (
            # Extract upper diagonal part (including diag)
            tf.matrix_band_part(self.LParamPost, 0, -1, name="UTriang_LParamPost")
            # Remove diagonal
            - tf.diag(tf.diag_part(self.LParamPost))
            # Add exp of original diagonal
            + tf.diag(tf.exp(tf.diag_part(self.LParamPost)))
        )
        # (M,M)
        LtL = tf.matmul(LParamPost_tri, LParamPost_tri, transpose_a=True, name="LtL")
        # (M,M)
        self.covCavityInv = self.KzzInv + LtL * np.array(
            self.n_points - self.set_for_training
        ) / np.array(self.n_points)

        # (M,M)
        self.covCavity = tf.cholesky_solve(
            tf.cholesky(self.covCavityInv, name="covCavity"), tf.eye(M, dtype=self.dtype)
        )
        # self.covCavity = tf.matrix_inverse(self.covCavityInv, name="covCavity")
        # self.mParamPost = tf.tile(self.mParamPost[None,:,:], [S,1,1])
        self.meanCavity = tf.matmul(
            self.covCavity,
            np.array(self.n_points - self.set_for_training)
            / np.array(self.n_points)
            * self.mParamPost
            + tf.matmul(self.KzzInv, self.mean_function_z),
        )

        self.KzzInvcovCavity = tf.matmul(
            self.KzzInv, self.covCavity, name="KzzInvcovCavity"
        )
        # A
        self.KzzInvmeanCavity = tf.matmul(
            self.KzzInv, self.meanCavity, name="KzzInvmeanCavity"
        )

        # (M,M)
        self.covPosteriorInv = self.KzzInv + LtL

        self.covPosterior = tf.cholesky_solve(
            tf.cholesky(self.covPosteriorInv), tf.eye(M, dtype=self.dtype)
        )
        # self.covPosterior = tf.matrix_inverse(self.covPosteriorInv, name='covPosterior')
        # (M,1)
        self.meanPosterior = tf.matmul(
            self.covPosterior,
            self.mParamPost + tf.matmul(self.KzzInv, self.mean_function_z),
            name="meanPosterior",
        )

        # Mean of the dist that we want to sample from.
        # We add the input of the node to the output as done in Neural Nets (Skip Layer).
        # self.mean_function_x = XW
        # Size (S,N,1)
        self.output_means = (
            tf.matmul(
                tf.matmul(
                    self.Kxz,
                    self.KzzInv * tf.ones([S, M, 1], dtype=self.dtype),
                    name="output_means",
                ),
                (self.meanCavity - self.mean_function_z)
                * tf.ones([S, M, 1], dtype=self.dtype),
            )
            + self.mean_function_x
        )

        # Compute the output vars
        # (S, M, M)
        self.B = tf.matmul(self.KzzInvcovCavity, self.KzzInv) - self.KzzInv * tf.ones(
            [S, M, M], dtype=self.dtype
        )
        # tf.exp(self.lsf) is the kernel Kxx
        v_out = tf.exp(self.lsf) + tf.matmul(
            self.Kxz * tf.matmul(self.Kxz, self.B, name="kzzB"),
            tf.ones([S, M, 1], dtype=self.dtype),
            name="matMulVout",
        )

        # Variance of the dist that we want to sample from
        # Size (S,N,1)
        self.output_vars = tf.abs(v_out)

    def get_params(self):
        """
            Returns the trainable variables of the GP node
        """
        # TODO: The list in the layer level (gp layer) could be converted
        # to a set. That way we could return everything without checks as sets don't allow
        # duplicates it wouldn't be a problem
        params = [self.mParamPost, self.LParamPost]
        if self.shared_layer_tf_variables.get("lls") is None:
            params.append(self.lls)
        if self.shared_layer_tf_variables.get("lsf") is None:
            params.append(self.lsf)
        if self.shared_layer_tf_variables.get("Z") is None:
            params.append(self.Z)
        return params

    ##
    # The next functions compute the log normalizer of each distribution (needed for energy computation)
    #
    # (1,1)
    def log_normalizer_cavity(self):
        assert (
            self.covCavity is not None
            and self.meanCavity is not None
            and self.covCavityInv is not None
        )

        return (
            0.5 * self.n_inducing_points * np.log(2 * np.pi)
            + 0.5 * self.log_det(self.covCavity)
            + 0.5
            * tf.matmul(
                tf.matmul(self.meanCavity, self.covCavityInv, transpose_a=True),
                self.meanCavity,
            )
        )

    # (1,)
    def log_normalizer_prior(self):
        assert self.KzzInv is not None
        return (
            0.5 * self.n_inducing_points * np.log(2 * np.pi)
            - 0.5 * self.log_det(self.KzzInv)
            + 0.5
            * tf.matmul(
                tf.matmul(self.mean_function_z, self.KzzInv, transpose_a=True),
                self.mean_function_z,
            )
        )

    # (1,1)
    def log_normalizer_posterior(self):
        assert (
            self.covPosterior is not None
            and self.meanPosterior is not None
            and self.covPosteriorInv is not None
        )
        # The first part is size (S, ), and we expect (S, 1, 1)
        return (
            0.5 * self.n_inducing_points * np.log(2 * np.pi)
            + 0.5 * self.log_det(self.covPosterior)
            + 0.5
            * tf.matmul(
                tf.matmul(self.meanPosterior, self.covPosteriorInv, transpose_a=True),
                self.meanPosterior,
            )
        )

    ##
    # We return the contribution to the energy of the node
    # (See last Eq. of Sec. 4 in http://arxiv.org/pdf/1602.04133.pdf v1)
    # (1,1)
    def get_node_contribution_to_energy(self):
        assert (
            self.n_points is not None
            and self.covCavity is not None
            and self.covPosterior is not None
        )

        logZpost = self.log_normalizer_posterior()
        logZprior = self.log_normalizer_prior()
        logZcav = self.log_normalizer_cavity()
        # We multiply by the minibatch size and normalize terms according to the total number of points (n_points)
        return (
            (logZcav - logZpost) + logZpost / self.n_points - logZprior / self.n_points
        ) * tf.cast(tf.shape(self.input_means)[1], self.dtype)

    # These methods sets the inducing points to be a random subset of the inputs (we should receive more
    # inputs than inducing points), the length scales are set to the mean of the euclidean distance
    def calculate_median(self, v):
        v = tf.reshape(v, [-1])
        m = tf.shape(v)[0] // 2
        return tf.nn.top_k(v, m).values[m - 1]

    def log_det(self, x):
        # TODO: Check: https://stackoverflow.com/questions/44194063/calculate-log-of-determinant-in-tensorflow-when-determinant-overflows-underflows
        # x -> (S, M, M)
        return tf.constant(2.0, shape=[1], dtype=self.dtype) * tf.reduce_sum(
            tf.log(tf.diag_part(tf.cholesky(x, name="CholeskyLogDet")))
        )

