##
# This class represents an output node that will contain the targets for a classification problem
# It plays the role of a probit function. The lables are either 1 or -1
#
from dgp_aepmcm.nodes.base_node import BaseNode


class OutputNodeBase(BaseNode):
    def __init__(self, y_train_tf, y_test_tf, n_samples):
        BaseNode.__init__(self)
        self.y_test_tf = y_test_tf
        self.y_train_tf = y_train_tf
        self.n_samples = n_samples
        self.n_samples_to_propagate = None

    def get_output(self):
        raise Exception("This is an output node, does not generate any output.")

    def get_predicted_values(self):
        raise Exception("Not implemented")
