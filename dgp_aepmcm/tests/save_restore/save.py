import os

import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.metrics import mean_squared_error

from dgp_aepmcm.gp_network import DGPNetwork


@np.vectorize
def f(x):
    return np.sin(2 * x)


N = 50
x = np.linspace(-2, 2, N)
y = f(x)
x = x[:, None]
y = y[:, None]

# DGP model variables
# Number of inducing points
M = 50
D = x.shape[-1]
# Maximun of epochs for training
max_epochs = 50  # 150
learning_rate = 0.01
minibatch_size = 500
# Inducing points locations
Z = kmeans2(x, M, minit="points")[0]
noise_val = 1e-5

model_aepmcm = DGPNetwork(x, y, inducing_points=Z, show_debug_info=True, dtype=np.float64)

model_aepmcm.add_input_layer()
# This method always assume a mean function for the prior p(u) = N(u| m(x), Kzz)
# with m(x) = X W
# For this example we disable the mean function for the prior so we set W to 0.
model_aepmcm.add_gp_layer(M, 3)  # W=np.zeros((D, 3)))
# model_aepmcm.add_noise_layer(noise_val)
model_aepmcm.add_gp_layer(M, 3)
# model_aepmcm.add_noise_layer(noise_val)
model_aepmcm.add_gp_layer(M, 1)
# model_aepmcm.add_noise_layer(noise_val)
model_aepmcm.add_output_layer_regression()

model_aepmcm.train_via_adam(
    max_epochs=max_epochs, minibatch_size=minibatch_size, learning_rate=learning_rate
)

print("Finished training")
directory = os.path.dirname(os.path.abspath(__file__))
model_aepmcm.save_model(f"{directory}/saved_model", "test_save")

y_pred, uncert = model_aepmcm.predict(x)

rmse_train = mean_squared_error(y, y_pred)
# recall_dgp = recall_score(y_test, labels_aepmcm)
print(f"Rmse Train: {rmse_train}")
# print(f"Recall: {recall_dgp}")
