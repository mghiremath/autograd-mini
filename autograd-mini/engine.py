import math
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

class Value:
    def __init__(self, data, _children=(), _op='', label = ''):
        self.data = data
        self.grad=0.0 # variable grad initialized as 0  - change w.r.to previous nodes value, usually weights
        self._backward = lambda: None  #default empty function
        self._prev = set(_children)
        self._op = _op
        self.label=label
        
    def __repr__(self):
        return f"Value(data={self.data})"
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
        output._backward = _backward
        return output

    def __radd__(self, other):  # for sum()
        return self + other


    def __sub__(self, other): # self - other
        return self + (-other)

    def __neg__(self):  # -self
        return self * -1

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward
        return output

    def __pow__(self, other): 
        assert isinstance(other, (int, float)), "only supported int/float powers for now"
        output = Value(self.data ** other, (self, ), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other-1)) * output.grad
        output._backward = _backward
        return output
    
    def __rmul__(self, other): # other* self
        return self*other
    
    def __truediv__(self, other): # self/other
        return self * other ** -1
        
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        output = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * output.grad
        output._backward = _backward
        return output
    
    def exp(self):
        x = self.data
        output = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += output.data * output.grad 
        output._backward = _backward
        return output
        
    def backward(self):

        topo=[]
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
