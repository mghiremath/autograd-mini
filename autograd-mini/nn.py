# Building neural net and library on top of it(MLP-Multi layer perceptron)
import random
from engine import Value
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        #W * X + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), start=Value(0.0))  #creates a zip iterator that pairs elements from two sequences
        output = act + self.b
        return output.tanh()
    
    def parameters(self):
        return self.w + [self.b]

# Layer of neurons
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def parameters(self):
        # params = []
        # for neuron in self.neurons:
        #     ps = neuron.parameters()
        #     params.extend(ps)
        # return params
        return [p for neuron in self.neurons for p in neuron.parameters()]
        
        
# MLP - Multi Layer Perceptron
class MLP:
    def __init__(self, nin, nouts):  # nin`: Number of inputs to the network | `nouts`: A list where each element represents the number of neurons in each layer
        sz = [nin] + nouts     #Creates a list that starts with the input size followed by the sizes of each layer. 
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]  #Creates the network layers:
                                                                           #- Uses a list comprehension to create Layer objects
                                                                           #- Each Layer connects the output of the previous layer to the input of the next layer
                                                                           #- The dimensions are determined by consecutive pairs in the `sz` list
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)  #Passes the input through each layer sequentially. The output of each layer becomes the input to the next layer
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

