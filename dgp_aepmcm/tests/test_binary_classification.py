from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import logsumexp, erf
from scipy.stats import norm

from dgp_aepmcm.nodes.output_node_classification import OutputNodeClassification
from scipy.stats import norm

# Params indicate with and without variance bias
@pytest.fixture(params=[True, False], ids=["norm_cdf_likelihood", "heavyside"])
def binary_classification_variables(request, default_common_variables_dgp):
    n_classes = 2
    S = default_common_variables_dgp["S"]
    use_norm_cdf = request.param
    if use_norm_cdf:
        variance_bias = tf.constant(1.0, default_common_variables_dgp["dtype"])
    else:
        variance_bias = tf.constant(0.0, default_common_variables_dgp["dtype"])
    normal_means = np.tile(default_common_variables_dgp["mu"][None, :, None], [S, 1, 1])
    normal_vars = np.tile(default_common_variables_dgp["var"][None, :, None], [S, 1, 1])
    # NOTE: We could set y_train to a np.argmax(mu, 1) so probabilities are bigger and
    # there is less chance of a test failing due to small probs.
    variables = {
        "n_classes": n_classes,
        "y_train": np.random.choice([-1, 1], size=(default_common_variables_dgp["N"], 1)),
        "y_train_tf": tf.placeholder(tf.int32, name="y_training", shape=[None, 1]),
        "y_test_tf": tf.placeholder(tf.int32, name="y_test", shape=[None, 1]),
        "input_means": normal_means,
        "input_vars": normal_vars,
        "variance_bias": variance_bias,
        "use_norm_cdf": use_norm_cdf,
    }
    default_common_variables_dgp.update(variables)
    return default_common_variables_dgp


def test_binary_classification_logz(binary_classification_variables):
    out_node = OutputNodeClassification(
        binary_classification_variables["y_train_tf"],
        binary_classification_variables["y_test_tf"],
        binary_classification_variables["S"],
        binary_classification_variables["variance_bias"],
        binary_classification_variables["dtype"],
    )
    out_node.get_input = MagicMock(
        return_value=(
            binary_classification_variables["input_means"],
            binary_classification_variables["input_vars"],
        )
    )
    excpected_log_z = log_z_by_quadrature_for_binary_classification(
        binary_classification_variables["y_train"],
        binary_classification_variables["mu"],
        binary_classification_variables["var"],
        binary_classification_variables["use_norm_cdf"],
    )
    log_z_tf = out_node.get_logz()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        log_z_node = sess.run(
            log_z_tf,
            feed_dict={
                binary_classification_variables[
                    "y_train_tf"
                ]: binary_classification_variables["y_train"]
            },
        )
    assert_allclose(
        log_z_node,
        excpected_log_z,
        atol=binary_classification_variables["tolerance"],
        err_msg=f"Binary classification log z error",
    )


def heavyside(x):
    return 1.0 if x >= 0 else 0.0


def log_z_by_quadrature_for_binary_classification(y_train, mean, var, use_norm_cdf):
    if use_norm_cdf:
        # use cdf likelihood
        likelihood = norm.cdf
    else:
        # Use heavyside (step likelihood)
        likelihood = heavyside

    expected_log_z_value = []
    for i in range(mean.shape[0]):
        m = mean[i]
        s = var[i] ** 0.5
        y = y_train[i, 0]
        expected_log_z_value.append(
            np.log(
                quad(lambda x: likelihood(y * x) * norm.pdf(x, m, s), -np.inf, np.inf)[0]
            )
        )
    return np.array(expected_log_z_value)[:, None]
