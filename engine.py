import numpy as np
DEFAULT_DTYPE = np.float32

# _backward() for each operation overrides the _backward() method for the output node that operation created
# When called, _backward() will find the local derivative at that step and multiply it by the incoming gradient
# Flow from loss -> leaves, so gradient at each step is dLoss/dstepN * dstepN/dstepN-1
# Each child calculates its parent's gradient, so gradient is accumulated and stored w/ += in its parent's .grad
# dLoss/dStepN is calculated by step N + 1's _backward(), thats how it knows the incoming gradient
# Info needed for the local derivative _backward() calculates is cached during the forward pass

# NOTE: Each step has a Tensor. Outside _backward(), self.grad is that Tensor's gradient. Inside _backward(), out.grad is that Tensor's gradient.
# self.grad is now the gradient of the Tensor that created it (previous step). THIS is what allows _backward() to find the gradient of a child's parent
# So when _backward does self.grad += out.grad (with some operation), it's setting the previous Tensor's self.grad using the current Tensor's self.grad calling it out.grad

class Tensor:
    def __init__(self, data, _creators=None, _op=None):
        self.data = np.asarray(data, dtype=DEFAULT_DTYPE)
        self.grad = np.zeros_like(self.data) # Gradient of this tensor
        self._creators = _creators if _creators is not None else []
        self._op = _op
        self._backward = lambda: None

    # Every operation stores the output, so the incoming gradient is just that output's gradient (out.grad, child informs parent gradient)
    def __add__(self, other):
        out = Tensor(self.data + other.data, _creators=[self, other], _op="add")
        
        # Local derivative is 1, just passes gradient through
        # A broadcast tensor was effectively reused many times in the forward pass,
        # so its gradient is the SUM over those uses — we sum back down to the original shape, gradient must match original shape
        def _backward():
            grad_self = out.grad
            grad_other = out.grad
            
            # Collapse any extra leading dimensions (e.g., Sequence Length)
            while len(grad_self.shape) > len(self.data.shape):
                grad_self = np.sum(grad_self, axis=0)
            
            # Collapse any dimensions that were originally size 1 (e.g., Batch)
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad_self = np.sum(grad_self, axis=i, keepdims=True)
                    
            # Repeat for 'other'
            while len(grad_other.shape) > len(other.data.shape):
                grad_other = np.sum(grad_other, axis=0)
                
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad_other = np.sum(grad_other, axis=i, keepdims=True)
            
            # Feed back the shape-altered gradients to the gradients of self and other
            self.grad += grad_self
            other.grad += grad_other
        
        # Set the output's backward placeholder to this custom backward method
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, _creators=[self, other], _op="matmul")
        
        # Involves operand transpose
        def _backward():
            # Safely swap ONLY the last two dimensions (matrix transpose), ignoring Batch dims
            def swap_inner(arr):
                return np.swapaxes(arr, -1, -2) if arr.ndim >= 2 else arr.T
                
            grad_self = out.grad @ swap_inner(other.data)
            grad_other = swap_inner(self.data) @ out.grad
            
            # Un-broadcast grad_self (Collapse extra dimensions)
            while len(grad_self.shape) > len(self.data.shape):
                grad_self = np.sum(grad_self, axis=0)
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad_self = np.sum(grad_self, axis=i, keepdims=True)
                    
            # Un-broadcast grad_other (Crucial for updating 2D weights with 3D sequences)
            while len(grad_other.shape) > len(other.data.shape):
                grad_other = np.sum(grad_other, axis=0)
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad_other = np.sum(grad_other, axis=i, keepdims=True)
            
            # Feed back the shape-altered gradients to the gradients of self and other
            self.grad += grad_self
            other.grad += grad_other
        
        out._backward = _backward
        return out

    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, _creators=[self], _op="relu")
        def _backward():
            # Pass gradient only where the input was positive (ReLU derivative is 1 there, 0 elsewhere)
            self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def sigmoid(self):
        # Clip to a shortened range to prevent NaN/warnings
        # Costs nothing in accuracy, sigmoid is practically 0 or 1 far before -500 or 500
        clipped = np.clip(self.data, -500, 500)
        s = 1.0 / (1.0 + np.exp(-clipped))

        out = Tensor(s, _creators=[self], _op="sigmoid")
        def _backward():
            self.grad += out.grad * out.data * (1.0 - out.data)
        out._backward = _backward
        return out

    def backward(self):
        # Holds tensors in topological order
        topo = []
        # Set to ensure each tensor is processed exactly once
        visited = set()

        # DFS method, for every unvisited node add all its parents before adding the node itself
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for creator in node._creators:
                    build_topo(creator)
                topo.append(node)
        build_topo(self)
        
        # Seed the gradient w/ 1, gradient of the loss w/ respect to itself is always 1
        self.grad = np.ones_like(self.data)

        # Backward pass (loss (top) -> leaves (bottom))
        # Ex. Loss = relu(Step N + 1) (so there's a tensor with Loss as the data, N + 1 as the creator)
        # Loss has self.grad = 1, its own _backward() treats loss as the output so out.grad = 1, and loss's backward() finds dLoss/dStepN + 1
        # This dLoss/dStepN + 1 is stored in self.grad (+=), but this self.grad refers to N + 1's tensor, NOT loss's tensor
        # Loss's backward in this example would be relu, allowing it to find how relu of Step N + 1 affects Loss i.e dLoss/dStepN + 1
        for node in reversed(topo):
            node._backward()
    
    # Turns a vector of raw scores (logits) into a probability distribution
    def softmax(self, axis=-1):
        # Stable forward pass
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
        
        out = Tensor(probs, _creators=[self], _op="softmax")
        
        def _backward():
            sum_dp = np.sum(out.grad * probs, axis=axis, keepdims=True)
            self.grad += probs * (out.grad - sum_dp)
            
        out._backward = _backward
        return out

    # Defines the official string representation of the object
    # Stores dimensions and operators
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op if self._op else 'Leaf'})"

# A class that:
# - May hold learnable parameters and/or other Modules
# - Defines a forward computation
# Can be anything from a single layer to the entire model
class Module:
    # Collects every learnable Tensor found in this module
    def parameters(self):
        params = []
        for attr_val in self.__dict__.values():
            # If its a bare Tensor, add it to the list
            if isinstance(attr_val, Tensor):
                params.append(attr_val)

            # If its a Module, recurse and get ITS parameters
            elif isinstance(attr_val, Module):
                params.extend(attr_val.parameters())
            
            # Do the last two steps on a list of layers (searching for Modules or Tensors)
            elif isinstance(attr_val, list):
                for item in attr_val:
                    if isinstance(item, Tensor):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    # *args and **kwargs means accept any arguments at all since layers take different inputs
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    # Convenience method, turns layer.forward(x) into layer(x)
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr # Learning Rate

    # Clears (zeroes) old gradient
    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.grad)

    def step(self):
        for p in self.parameters:
            # Move down the gradient (-=) by gradient * learning rate
            p.data -= self.lr * p.grad