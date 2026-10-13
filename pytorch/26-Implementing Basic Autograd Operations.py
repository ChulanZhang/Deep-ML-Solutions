class Value:
    """
    A scalar value that supports basic operations and automatic differentiation (autograd).
    Similar to a simplied version of micrograd.
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Derivative of addition is 1, so gradient passes through equally
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Derivative of product rule: d(x*y)/dx = y, d(x*y)/dy = x
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')

        def _backward():
            # Derivative of ReLU is 1 if data > 0, else 0
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        
        return out

    def backward(self):
        # 1. Topological sort to order the nodes for backpropagation
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)

        # 2. Seed the initial gradient to 1.0 (dz/dz = 1)
        self.grad = 1.0
        
        # 3. Apply the chain rule moving backwards through the graph
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
