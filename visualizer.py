import matplotlib.pyplot as plt 
import torch 

class NNVisulizer:
    POS_COLOR = "blue"
    NEG_COLOR = "red"
    MIN_WEIGHT = -1
    MAX_WEIGHT = 1
    MIN_NODE = -1
    MAX_NODE = 1 

    def __init__(self):
        self.weights = []
        self.nodes = []

    #TODO add args capablity to this function 
    #adds weights via a list or single weight 
    def add_weight_layer(self, weights):
        #TODO add safegaurd for list of tensors
        if type(weights) is list or type(weights) is torch.tensor:
            self.weights.append(weights)

    def clear_weights(self):
        self.weights = []


    def add_node_layer(self, nodes):
        pass
 
    def clear_nodes(self):
        self.nodes = []

    def render_no_data(self):
        pass

    def render_with_data(self):
        pass
