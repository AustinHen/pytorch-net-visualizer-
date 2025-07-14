This tool creates visual representations of neural networks (as pictured below)

<img width="683" height="703" alt="demo-1" src="https://github.com/user-attachments/assets/e46c8f49-87b0-44cf-899e-e244c6188223" />

<img width="683" height="703" alt="demo-2" src="https://github.com/user-attachments/assets/f9224eb2-eb85-47e8-8ac4-055529f35122" />

Produced graphics can either be "static": just showing the topology of the network or "reactive": showing data being passed through the network (indicated through the nodes changing color). 

When called without data, line thickness is used to denote the magnitude of each weight and color (red or blue) is used to show if the weight is positive or negative. 

When called with data, line opacity is used to display the activation of each weight and color is used to show the intermediate value at each node. 

The style of the graph can be changed very easily.

*Note: this tool has only been tested with nn.Linear based networks. I intend to add support for other network types in the future.*

## Usage
say you have the very basic neural network defined below  
```.py
layers = []
layers.append(nn.Linear(10, 10)) 
layers.append(nn.Linear(10, 5)) 

def forward(layers, inp):
    for layer in layers:
        if layer == layers[-1]:
            #usually something diff for last layer im just being lazy
            return F.relu(layer(inp))
        else:
            imp = F.relu(layer(inp))
```

To render this without data you would 
1) create an instance of the NNVisulizer class 
2) feed all the layers into the class using the add_weight_layer method 
3) call render 

So the example above becomes: 

```.py
renderer = visualizer.NNVisulizer() 
layers = [] 
layers.append(nn.Linear(10, 10)) 
layers.append(nn.Linear(10, 5)) 
for layer in layers:
	renderer.add_weight_layer(layers)
render()

def forward(layers, inp):
    for layer in layers:
        if layer == layers[-1]:
            #usually something diff for last layer im just being lazy
            return F.relu(layer(inp))
        else:
            imp = F.relu(layer(inp))
```

note the forward method remains unchanged

To draw a network with data you would take the code above and add a new forward method that calls add_node_layer (ik poorly named) with every intermediate product. 

The example above becomes:
```.py
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
```

---
I made this when working with the NEAT algorithm. In the future, if I find myself wanting better visuals for unsupported network types, I will add them. 
