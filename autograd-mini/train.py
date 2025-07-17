from graphviz import Digraph
# train.py
from nn import MLP
from engine import Value

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges
def draw_dot(root):
    dot = Digraph (format = 'svg', graph_attr = {'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape = 'record')
        if n._op:
            # if this value is a result of some opertation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot



x = [2.0, 3.0, -1.0]

n = MLP(3, [4, 4, 1])
n(x) #invokes __call__
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]   # desired targets

# Show predictions BEFORE training
print("=== Predictions BEFORE training ===")
ypred_before = [n(x) for x in xs]
for i, y in enumerate(ypred_before):
    print(f"Input {i}: predicted = {y.data:.4f}, target = {ys[i]}")

for k in range(100):
    # Forward pass
    ypred = [n(x) for x in xs]

    # Compute loss = sum of squared differences
    loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)), start=Value(0.0))

    # Backpropagate - backward pass
    for p in n.parameters():
        p.grad = 0.0  # Do not forget to zero grad before every backward pass
    loss.backward()

    # Update
    for p in n.parameters():
        p.data -= 0.05 * p.grad  # minimize the loss  - opposite to the direction of increasing the loss function

    # View loss
    print(k, loss.data)

    # Print loss every 100 iterations
    if k % 100 == 0:
        print(f"Epoch {k}: Loss = {loss.data:.4f}")

# Show predictions AFTER training
print("\n=== Predictions AFTER training ===")
ypred_after = [n(x) for x in xs]
for i, y in enumerate(ypred_after):
    print(f"Input {i}: predicted = {y.data:.4f}, target = {ys[i]}")
