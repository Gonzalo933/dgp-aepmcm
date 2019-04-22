import numpy as np
import tensorflow as tf


def compute_kernel(lls, lsf, X, X2=None, dtype=np.float32):
    """
    This function computes the RBF covariance matrix for the GP
    """
    if X2 is None:
        X2 = X
        jitter = 1e-5
        white_noise = jitter * tf.eye(tf.shape(X)[0], dtype=dtype)
    else:
        white_noise = 0.0

    X = X / tf.sqrt(tf.exp(lls))
    X2 = X2 / tf.sqrt(tf.exp(lls))
    value = tf.expand_dims(tf.reduce_sum(tf.square(X), 1), 1)
    value2 = tf.expand_dims(tf.reduce_sum(tf.square(X2), 1), 1)
    distance = value - 2 * tf.matmul(X, tf.transpose(X2)) + tf.transpose(value2)

    return tf.exp(lsf) * tf.exp(-0.5 * distance) + white_noise


def compute_kernel_numpy(lls, lsf, x, z=None):

    ls = np.exp(lls)
    sf = np.exp(lsf)

    if z is None:
        z = x.copy()
        jitter = 1e-5
        white_noise = jitter * np.eye(x.shape[0])
    else:
        white_noise = 0.0

    if x.ndim == 1:
        x = x[None, :]

    if z.ndim == 1:
        z = z[None, :]

    lsre = np.outer(np.ones(x.shape[0]), ls)
    r2 = (
        np.outer(np.sum(x * x / lsre, 1), np.ones(z.shape[0]))
        - 2 * np.dot(x / lsre, z.T)
        + np.dot(1.0 / lsre, z.T ** 2)
    )
    k = sf * np.exp(-0.5 * r2) + white_noise

    return k
