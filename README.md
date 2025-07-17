# Micrograd Clone in Python 🔥

This is a tiny neural net + autograd engine inspired by [micrograd](https://github.com/karpathy/micrograd). It supports scalar reverse-mode autodiff and builds a small MLP from scratch.

## Features
- Scalar-based autograd system (`Value` class)
- Neural network: `Neuron`, `Layer`, `MLP`
- Backpropagation from scratch
- Simple training loop

##  Graph Visualization

You can visualize the computation graph using Graphviz:

```python
from graphviz_utils import draw_dot
dot = draw_dot(loss)
dot.render("graph", view=True)

## Usage
```bash
python train.py
