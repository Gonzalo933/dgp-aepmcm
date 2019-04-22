from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_allclose

import tensorflow as tf
from dgp_aepmcm.nodes.output_node_multiclass import OutputNodeMulticlass
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm

# Params indicate with and without variance bias
@pytest.fixture(
    params=[True, False], ids=["with noise in labels", "without noise in labes"]
)
def multiclass_classification_variables(request, default_common_variables_dgp):
    n_classes = 3
    S = default_common_variables_dgp["S"]
    prob_wrong_label = logit(1e-3)
    noise = request.param
    N = default_common_variables_dgp["N"]
    mu = np.random.uniform(-3, 3, size=(N, n_classes))
    var = np.tile(0.1, (N, n_classes))
    normal_means = np.tile(mu[None, :, :], [S, 1, 1])
    normal_vars = np.tile(var[None, :, :], [S, 1, 1])
    # NOTE: if we set y_train randomly i.e np.random.randint(0, n_classes, size=(N, 1))
    # then, when making tests with noise=False, they will fail,
    # because probabilities will be too small.
    variables = {
        "n_classes": n_classes,
        "y_train": np.argmax(mu, 1)[:, None],
        "y_train_tf": tf.placeholder(tf.int32, name="y_training", shape=[None, 1]),
        "y_test_tf": tf.placeholder(tf.int32, name="y_test", shape=[None, 1]),
        "input_means": normal_means,
        "input_vars": normal_vars,
        "prob_wrong_label": prob_wrong_label,
        "sigmoid_prob_wrong_label": 1 / (1 + np.exp(-prob_wrong_label)),
        "noise": noise,
        "mu": mu,
        "var": var,
    }
    default_common_variables_dgp.update(variables)
    return default_common_variables_dgp


def logit(p):
    return np.log(p / (1.0 - p))


def test_multiclass_classification_predict_probs_sum_one(
    multiclass_classification_variables
):
    # Tests that \sum_{i=1}^{n_classes} p(y=i | x) == 1
    noise = multiclass_classification_variables["noise"]
    out_node = OutputNodeMulticlass(
        multiclass_classification_variables["y_train_tf"],
        multiclass_classification_variables["y_test_tf"],
        multiclass_classification_variables["S"],
        multiclass_classification_variables["n_classes"],
        noise,
        True,
        multiclass_classification_variables["dtype"],
    )
    out_node.get_input = MagicMock(
        return_value=(
            multiclass_classification_variables["input_means"],
            multiclass_classification_variables["input_vars"],
        )
    )
    predict_tf = out_node.get_predicted_values()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        predicted_vals, probs_dgp_network = sess.run(
            predict_tf,
            feed_dict={
                multiclass_classification_variables[
                    "y_test_tf"
                ]: multiclass_classification_variables["y_train"]
            },
        )
    assert_allclose(
        np.sum(probs_dgp_network, 1),
        np.ones(multiclass_classification_variables["N"]),
        atol=multiclass_classification_variables["tolerance"],
        err_msg=f"Output node probs with noise={noise} doesn't sum to one",
    )


def test_multiclass_classification_quad_probs_sum_one(
    multiclass_classification_variables
):
    noise = multiclass_classification_variables["noise"]
    probs = []
    for k in range(multiclass_classification_variables["n_classes"]):
        probs.append(
            np.exp(
                log_z_by_quadrature(
                    multiclass_classification_variables["mu"],
                    multiclass_classification_variables["var"],
                    multiclass_classification_variables["y_train"],
                    multiclass_classification_variables["n_classes"],
                    noise,
                    multiclass_classification_variables["sigmoid_prob_wrong_label"],
                    k,
                )
            )
        )
    assert_allclose(
        np.sum(probs, 0),
        np.ones(multiclass_classification_variables["N"]),
        atol=multiclass_classification_variables["tolerance"],
        err_msg=f"Quadrature with noise={noise} doesn't sum to one",
    )


def test_multiclass_classification_logz(multiclass_classification_variables):
    noise = multiclass_classification_variables["noise"]
    out_node = OutputNodeMulticlass(
        multiclass_classification_variables["y_train_tf"],
        multiclass_classification_variables["y_test_tf"],
        multiclass_classification_variables["S"],
        multiclass_classification_variables["n_classes"],
        noise,
        True,
        multiclass_classification_variables["dtype"],
    )
    out_node.get_input = MagicMock(
        return_value=(
            multiclass_classification_variables["input_means"],
            multiclass_classification_variables["input_vars"],
        )
    )
    log_z_tf = out_node.get_logz()
    expected_log_z_value = log_z_by_quadrature(
        multiclass_classification_variables["mu"],
        multiclass_classification_variables["var"],
        multiclass_classification_variables["y_train"],
        multiclass_classification_variables["n_classes"],
        noise,
        multiclass_classification_variables["sigmoid_prob_wrong_label"],
    )
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        log_z = sess.run(
            log_z_tf,
            feed_dict={
                multiclass_classification_variables[
                    "y_train_tf"
                ]: multiclass_classification_variables["y_train"]
            },
        )[:, 0]
    assert_allclose(
        log_z,
        expected_log_z_value,
        atol=multiclass_classification_variables["tolerance"],
        err_msg=f"Error output node logz multiclass with noise={noise}",
    )


def log_z_by_quadrature(
    mu_classes,
    var_classes,
    y_train,
    n_classes,
    noise,
    sigmoid_prob_wrong_label,
    target_class=None,
):
    mean, var = (mu_classes, var_classes)
    possible_classes = set(range(n_classes))
    expected_values_quad = []
    y = y_train if target_class is None else np.ones_like(y_train) * target_class
    for i in range(mean.shape[0]):
        class_i = y[i, 0]
        mu_y, var_y = mean[i, class_i], var[i, class_i]
        indexes_other_classes = list(set(range(n_classes)) - set([y[i, 0]]))
        mu_rest_classes = mean[i, indexes_other_classes]
        var_rest_classes = var[i, indexes_other_classes]
        expected_log_z_value = quad(
            lambda x: calculate_cdf_prod(
                x,
                mu_rest_classes,
                var_rest_classes,
                sigmoid_prob_wrong_label,
                n_classes,
                noise=noise,
            )
            * norm.pdf(x, mu_y, np.sqrt(var_y)),
            -np.inf,
            np.inf,
        )[0]
        expected_values_quad.append(np.log(expected_log_z_value))
    return np.array(expected_values_quad)


def calculate_cdf_prod(
    x, mu_rest_classes, var_rest_classes, sigmoid_prob_wrong_label, n_classes, noise
):
    if noise:
        prod = np.prod(norm.cdf((x - mu_rest_classes) / np.sqrt(var_rest_classes)))
        return prod * (1 - sigmoid_prob_wrong_label) + (1 - prod) * (
            sigmoid_prob_wrong_label / (n_classes - 1)
        )
    else:
        return np.prod(norm.cdf((x - mu_rest_classes) / np.sqrt(var_rest_classes)))
