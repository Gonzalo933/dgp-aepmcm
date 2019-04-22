class BaseLayer:
    """Base class for all layers, contains common functions used by all layers of the network.

    """

    def __init__(self):
        self.node_list = []
        self.n_nodes = None
        self.n_samples = None
        self.stacked = False

    @property
    def n_nodes(self):
        return self.__n_nodes

    @n_nodes.setter
    def n_nodes(self, n_nodes):
        self.__n_nodes = n_nodes

    def add_node(self, node):
        """Adds a new (already created) node to the network. Note that this not connect the node

        Args:
            node (Object): Node to add to the network.
        """
        self.node_list.append(node)

    def get_node_list(self):
        """ Returns the list of nodes in the layer"""
        return self.node_list

    def forward_pass_computations(self):
        """ Do a forward pass through all the nodes in the layer """
        for node in self.get_node_list():
            node.forward_pass_computations()

    def stack_on_previous_layer(self, previous_layer):
        """ Connects a layer to the previous one

        Args:
            previous_layer (Object): Layer to connect to
        """
        nodes_previous_layer = previous_layer.get_node_list()
        for node in self.get_node_list():
            node.set_nodes_as_input(nodes_previous_layer)
        self.stacked = True

    def initialize_params_layer(self):
        """ This method initializes the parameters of the layer
            Should be overwritten (currently it is only used in Noise_layer and GP_layer) """
        pass

    def get_params(self):
        """ Returns the params of all the nodes in the layer """
        params = []
        for node in self.get_node_list():
            params = params + node.get_params()

        return params

    def get_layer_contribution_to_energy(self):
        """ Returns no contribution to the energy,
            layers that do contribute to energy should overwrite this method
        """
        return 0.0
