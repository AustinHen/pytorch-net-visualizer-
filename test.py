import visualizer
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F 

def run_all_tests():
    x = nn.Linear(28*28, 64).weight
    print(x[0,0])
    ##tester = visualizer.NNVisulizer()
    ##tester.render_no_data()
    test_2()

def test_1():
    print("running test 1") 
    tester = visualizer.NNVisulizer()
    tester.add_weight_layer(nn.Linear(10, 10).weight).add_weight_layer(nn.Linear(10, 5).weight)
    tester.render()

def test_2():
    layers = []
    tester = visualizer.NNVisulizer()
    layers.append(nn.Linear(10, 10)) 
    layers.append(nn.Linear(10, 5)) 
    for layer in layers:
        tester.add_weight_layer(layer.weight)
    inp = torch.randn((10))
    forward(layers, inp, tester)

def forward(layers, inp, v):
    v.add_node_layer(inp)
    for layer in layers:
        if layer == layers[-1]:
            #usually something diff for last layer im just being lazy
            inp = F.relu(layer(inp))
            v.add_node_layer(inp)
        else:
            imp = F.relu(layer(inp))
            v.add_node_layer(inp)
    v.render(True)
            
if __name__ == "__main__":
    run_all_tests()
