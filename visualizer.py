import matplotlib.pyplot as plt
import numpy as np
import torch 

class NNVisulizer:
    def __init__(self):
        self.weights = []
        self.nodes = []
        self.sizes = []

    def add_weight_layer(self, *args):
        for arg in args:
            if type(arg) == "list":
                self.weights.extends(arg)
            else:
                self.weights.append(arg)

        return self

    def clear_weights(self):
        self.weights = []
        return self

    def add_node_layer(self, nodes):
        self.nodes.append(nodes)
        return self
 
    def clear_nodes(self):
        self.nodes = []
        return self

    def render(self, with_data=False):
        if with_data:
            assert len(self.nodes) == 1 + len(self.weights), "incorrect number of node layers"

        #sets up mpl 
        fig, ax = plt.subplots(facecolor=".6")
        
        #gets size of nn / info to better display it
        self.update_sizes()
        center = self.get_center()

        #gets offsets
        offsets = []
        for size in self.sizes:
            offsets.append(center - size/2)

        #used to determine if a node should be displayed
        next_ingoing_sums = np.ones(self.sizes[0]).tolist()

        #plots
        for size_index, size in enumerate(self.sizes):
            x_cord = size_index + 1
            ingoing_sums = next_ingoing_sums
            for y_cord in range(size):
                offset = offsets[size_index]

                #plots nodes
                if ingoing_sums[y_cord] > 0:
                    _color = [0, 0, 1]
                    if with_data:
                        #this prob all could be a one liner im just not seeing it
                        node_val = float(self.nodes[0][y_cord])
                        c_val = node_val
                        c_val = -1 if c_val < -1 else c_val
                        c_val = 1 if c_val > 1 else c_val
                        #updates color
                        primary_index = 2 if abs(node_val) == node_val else 0
                        secondary_index = 0 if abs(node_val) == node_val else 2

                        _color[secondary_index] = 0
                        _color[primary_index] = abs(c_val)

                    ax.plot(x_cord, y_cord+offset, 'o', color=_color, markeredgewidth=10, zorder=2)

                #can skip rest if its the last layer
                if(size_index+1 == len(self.sizes)):
                    continue
                
                #plots outgoing weights 
                next_ingoing_sums = np.zeros(self.sizes[size_index+1]).tolist()
                next_offset = offsets[size_index+1]

                for w_index in range(self.sizes[size_index+1]): 
                    cur_weight_val = float(self.weights[size_index][w_index][y_cord])
                    print("weight val")
                    print(cur_weight_val)

                    next_ingoing_sums[w_index] += abs(cur_weight_val)

                    #draws line
                    xs = [size_index+1, size_index+2]
                    ys = [y_cord+offset, w_index+next_offset]

                    _color = 'r' if cur_weight_val < 0 else 'b'
                    _width = self.get_weight_line_width(cur_weight_val)

                    if not self.weight_is_cutoff(cur_weight_val):
                        plt.plot(xs, ys, color=_color, zorder=1, linewidth = _width)

        #shows plot 
        plt.show()

    def update_sizes(self):
        self.sizes = []
        for weight in self.weights:
            try:
                self.sizes.append(weight.size(dim=1))
            except:
                print("error -> weights likely incorrect format")
                exit(1)
        self.sizes.append(self.weights[-1].size(dim=0))

    #returns center + 1 of largest node layer -> used to center the plot
    def get_center(self):
        largest = max(self.sizes)
        return int(largest/2) + 1

    #used to determine if a weight should be displayed
    def weight_is_cutoff(self, w): 
        return w == 0

    #determines the line width given the weight
    def get_weight_line_width(self, w):
        return w*20

    




