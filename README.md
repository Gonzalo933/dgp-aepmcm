# DGP-AEPMCM
Code for the paper: "Deep Gaussian Processes using Expectation Propagation and Monte Carlo Methods"

## Requirements
- Python>=3.6
- Tensorflow>=1.12.0 (Although it may work with older versions). (**UPDATE: Using tensorflow 1.13 triggers some warnings.**)


## Installation
```shell
pip install --user .
```


## Known issues

- Sometimes the cholesky decomposition fails:
  - Increasing the jitter higher helps but at the cost of losing some accuracy in the network predictions (as it is noise added to the diagonal of the kernel between the inputs in each node).
  - Using float64 is also known to help. In future versions it will be possible to specify the precision of the floats used inside the network.