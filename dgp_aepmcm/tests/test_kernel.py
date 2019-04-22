import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

from dgp_aepmcm.kernel.gauss import compute_kernel, compute_kernel_numpy


class TestKernel:
    @classmethod
    def setup_class(self):
        tf.reset_default_graph()
        self.lls = np.array([np.log(np.random.randint(2, 10))], dtype=np.float64)
        self.lsf = np.array(np.log(np.random.randint(2, 10)), dtype=np.float64)
        self.N = np.random.randint(2, 10)
        self.x = np.random.normal(1, 2, (self.N, 1))

        self.x_tf = tf.placeholder(tf.float64, shape=self.x.shape, name="x")
        self.lls_tf = tf.placeholder(tf.float64, shape=self.lls.shape, name="lls")
        self.lsf_tf = tf.placeholder(tf.float64, shape=self.lsf.shape, name="lsf")
        self.tolerance = 1e-3

    def test_rbf_kernel(self):
        with tf.Session() as sess:
            result_tf = sess.run(
                compute_kernel(self.lls_tf, self.lsf_tf, self.x_tf, self.x_tf),
                feed_dict={
                    self.x_tf: self.x,
                    self.lls_tf: self.lls,
                    self.lsf_tf: self.lsf,
                },
            )
            result_numpy = compute_kernel_numpy(self.lls, self.lsf, self.x, self.x)
            assert_allclose(result_tf, result_numpy)
            assert_allclose(
                result_tf,
                result_numpy,
                atol=self.tolerance,
                err_msg="Error when computing kernel",
            )
