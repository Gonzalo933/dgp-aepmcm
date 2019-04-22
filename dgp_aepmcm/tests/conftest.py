import tensorflow as tf
import pytest
import numpy as np


@pytest.fixture
def default_common_variables_dgp():
    np.random.seed(5)
    tf.reset_default_graph()
    N = 10
    variables = {
        "N": N,
        "S": 1000,
        "mu": np.random.uniform(-3, 3, size=N),
        "var": np.repeat(0.1, N),
        "dtype": np.float64,
        "tolerance": 1e-3,
    }
    return variables
