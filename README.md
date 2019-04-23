# DGP-AEPMCM
Code for the paper: "Deep Gaussian Processes using Expectation Propagation and Monte Carlo Methods"

## Requirements
- Python>=3.6
- Tensorflow>=1.12.0 (Although it may work with older versions). (**UPDATE: Using tensorflow 1.13 triggers some warnings.**)


## Installation
```shell
pip install --user .
```

## Experiments
Datasets and splits for reproducing the experiments can be downloaded from here:
- [Regression datasets and splits.](https://gonzalohernandezmunoz.com/downloads/machine_learning/datasets_tfm.zip)

- [Binary classification datasets and splits.](https://gonzalohernandezmunoz.com/downloads/machine_learning/datasets_binary_classification.tar.gz)

## Docs
Code is partially documented, I'm working on it. Other documentation can be found here:
- [Notes on multiclass classification](dgp_aepmcm/nodes/latex/output_node_multiclass_info.pdf)
- [Master's thesis pdf](https://gonzalohernandezmunoz.com/downloads/machine_learning/Gonzalo_Hernandez_master_thesis.pdf)
- [Presentation](https://gonzalohernandezmunoz.com/downloads/machine_learning/Gonzalo_Hernandez_master_thesis_presentation.pdf)

## Known issues

- Sometimes the cholesky decomposition fails:
  - Increasing the jitter higher helps but at the cost of losing some accuracy in the network predictions (as it is noise added to the diagonal of the kernel between the inputs in each node).
  - Using float64 is also known to help. In future versions it will be possible to specify the precision of the floats used inside the network.

## Running tests

```shell
pytest -v dgp_aepmcm/tests/
```

Show warnings too:
```shell
pytest -r dgp_aepmcm/tests/
```
