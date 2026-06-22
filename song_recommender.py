import json
import glob
import os
import numpy as np

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
        self.data = np.asarray(data, dtype=np.float32)
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

# In/out_features describe the dimensions of the input and output
# X is the actual input
class LinearLayer(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Use He initialization to intialize weight to safe, random values (bias set to 0)
        # Values to be changed later during the learning portion (backprop) of the network
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * scale, _op="Weight")
        self.B = Tensor(np.zeros((1, out_features)), _op="Bias")

    def forward(self, X):
        return (X @ self.W) + self.B

# (Module) means the class inherits Module
class ReLU(Module):
    # Overrides Module's generic forward method, calls relu method on input tensor
    def forward(self, X): return X.relu()

class Sigmoid(Module):
    # Overrides Module's generic forward method, calls sigmoid method on input tensor
    def forward(self, X): return X.sigmoid()

# Allows for the creation of a resuable network architecture definition
# Takes in a list of Modules
# Ex. Sequential = ([LinearLayer(2,4), ReLU(), LinearLayer(4,1), Sigmoid()])
class Sequential(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

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

class Embedding(Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    Turns discrete integer IDs into dense continuous vectors.
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # Initialize a massive lookup table with small random numbers.
        # Shape: (Total unique songs in database, Number of latent features)
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim) * 0.1, _op="Embedding")

    # Takes a list of song IDs and returns each song's learnable feature vector
    def forward(self, indices):
        """
        indices: A NumPy array of integer IDs representing the sequence of songs.
        """
        # Ensure indices are integers for array lookup
        idx_array = np.asarray(indices, dtype=int)
        
        # Grab rows corresponding to indices, gets the embedding vectors for every song ID in indices
        out_data = self.weight.data[idx_array]
        
        # Wrap the looked-up vectors in a Tensor
        # Record that this output came from the embedding table to connect the lookup to the computation graph
        # so gradient can flow back to the table during backprop
        out = Tensor(out_data, _creators=[self.weight], _op="embed_lookup")

        # The method called by the Tensor (out) holding the looked up vectors
        # Takes incoming gradient out.grad (dL/d(looked-up vectors)) to find embedding table gradient
        def _backward():
            # Create a completely blank gradient matrix for the entire catalog
            grad_update = np.zeros_like(self.weight.data)
            
            # This handles 2D batches (multiple lists of indice lookups), turns them into one unified list
            flat_indices = idx_array.flatten()

            # Reshape gradient (number of lookups, number of features)
            # Remember forward returns out as the looked up vectors, so out.grad is the gradient of loss w/ respect
            # to each looked-up embedding vector
            flat_grad = out.grad.reshape(-1, self.weight.data.shape[-1])
            
            # Un-indexed Accumulation: Safely add the gradients ONLY to the rows 
            # that were actually accessed during this specific forward pass.
            # At each index k in flat_indices, do grad_update[flat_indices[k]] += flat_grad[k], sum to account for repeated songs
            np.add.at(grad_update, flat_indices, flat_grad)
            
            # Apply the sparse update to the master weight gradient
            self.weight.grad += grad_update

        out._backward = _backward
        return out

class SelfAttention(Module):
    """The mathematical heart of the Transformer."""
    # Why are QKV LinearLayers?
    # - Learnable: The weights inside get trained by backprop, so the network learns how to project a song into Q/K/V space
    # - Autograd for free: Calling self.W_query(X) builds the computation graph automatically
    def __init__(self, embed_dim):
        super().__init__()
        # Dot product the query and keys together to find the relevance of each song's key to each song's query
        # Take a softmax of all dot products, use them as weights to add their corresponding Value vectors
        # This is how the model learns "these songs belong together"
        self.W_query = LinearLayer(in_features=embed_dim, out_features=embed_dim)
        self.W_key = LinearLayer(in_features=embed_dim, out_features=embed_dim)

        # Song_Embedding @ W_value + B_value, a set of vectors representing the content each song contributes when attended to
        # NOTE: "A is attended to by B" means A's content (Value) is pulled into B's output, a song that's heavily attended to
        # means it was paid a lot of attention, its Value shows up strongly in other outputs
        self.W_value = LinearLayer(in_features=embed_dim, out_features=embed_dim)

        # Divides Q·K scores so they don't grow huge with dimension, keeping softmax smooth and gradients healthy
        self.scale = np.sqrt(embed_dim)

    def forward(self, X):
        # Takes in the set X of songs, creates Q,K,V (a collection of each song's Query, Key, Value vectors)
        Q = self.W_query(X)
        K = self.W_key(X)
        V = self.W_value(X)
        
        # Swap ONLY the last two dimensions (Sequence <-> Embed), leaving Batch alone.
        # Ex. Shape goes from (32, 3, 64) to (32, 64, 3)
        # Used to find dot product of Q and K, Q @ Kᵀ
        k_t_data = np.swapaxes(K.data, -1, -2)
        K_T = Tensor(k_t_data, _creators=[K], _op="transpose")
        
        def transpose_backward():
            # The exact reverse operation for the backward pass
            K.grad += np.swapaxes(K_T.grad, -1, -2)
        K_T._backward = transpose_backward

        # Find how much each song matches each song, scores[i][j] = song i's QUERY · song J's KEY = relevance of j to i
        scores = Q @ K_T 
        
        # Scale down scores, NOTE: NOT a Tensor operation so NOT tracked by Autograd, minor issue but it is slightly incorrect
        scores.data = scores.data / self.scale 

        # Find the weight of each Query Key pair where row 'i' is "how much song i attends to each song"
        # Relevance as probabilities, rows sum to 1
        attention_weights = scores.softmax(axis=-1)

        # Add the weighted sum of the Values, each song = weighted blend of Values
        contextualized_output = attention_weights @ V
        
        return contextualized_output # Same shape as the input, allows tranformer blocks to stack

# Rescales each vector so its values have mean 0 and standard deviation 1
class LayerNorm(Module):
    """Stabilizes the network by keeping vector numbers from exploding or collapsing."""
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Gamma and Beta allow for each vector (feature) to shift the rigid starting mean 0 / std 1
        # Ex. "Heavy guitar" feature could be more useful at a larger scale w/ smaller shift, so increase gamma decrease beta
        self.gamma = Tensor(np.ones(features), _op="LN_gamma")   # scale, starts at 1
        self.beta = Tensor(np.zeros(features), _op="LN_beta")    # shift, starts at 0

    def forward(self, X):
        # Calculate the mean and variance of each vector (feature)
        mean = np.mean(X.data, axis=-1, keepdims=True)
        var = np.var(X.data, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        
        # Normalize: (Value - Mean) / Standard Deviation
        x_hat = (X.data - mean) / std
        out_data = x_hat * self.gamma.data + self.beta.data # Scale + shift, done PER FEATURE

        out = Tensor(out_data, _creators=[X, self.gamma, self.beta], _op="layernorm")
        
        def _backward():
            g = out.grad

            # Gradients for the learnable params (sum over every axis EXCEPT features)
            reduce_axes = tuple(range(g.ndim - 1))
            self.beta.grad  += np.sum(g, axis=reduce_axes) # β just shifts -> grad is the summed upstream
            self.gamma.grad += np.sum(g * x_hat, axis=reduce_axes) # γ scales x_hat -> grad is summed g*x_hat

            # Gradient into X: pass through γ FIRST, then the same 3-term normalization backward
            g_xhat = g * self.gamma.data
            mean_g       = np.mean(g_xhat, axis=-1, keepdims=True)
            mean_g_xhat  = np.mean(g_xhat * x_hat, axis=-1, keepdims=True)
            X.grad += (g_xhat - mean_g - x_hat * mean_g_xhat) / std

        out._backward = _backward
        return out

# Inputs a tensor of shape (batch, seq_len, embed_dim) - a batch of song sequences, each song a vector of embed_dim features
# Outputs a tensor of the exact same shape and same sequence, but each song's vector has been refined (contextualized by other songs and individually transformed)
class TransformerBlock(Module):
    """A complete Transformer processing block (Attention + Add/Norm + FFN + Add/Norm)"""
    def __init__(self, embed_dim):
        super().__init__()
        # Gathers context
        self.attention = SelfAttention(embed_dim)
        self.norm1 = LayerNorm(embed_dim)
        
        # We expand the 64 dimensions to 256 to ask more "questions", then compress back to 64
        hidden_dim = embed_dim * 4  
        self.ffn_expand = LinearLayer(in_features=embed_dim, out_features=hidden_dim)

        # Adds non-linearity, necessary for solving problems linear functions cannot
        self.ffn_relu = ReLU()
        self.ffn_compress = LinearLayer(in_features=hidden_dim, out_features=embed_dim)
        
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, X):
        # Calculates attention, context is now a weighted blend of the other song's values per song
        context = self.attention(X)
        
        # Add new context (weighted blend of each song per song) + original songs
        # Preserves original info (context only adjusts it) and allows for backprop to flow through deep networks
        # Without it, gradient is F'_{L-1} · ... · F'_2 · F'_1, if "F'"s are < 1 which is typical, product (gradient) shrinks to 0
        # By adding X, the derivative is now (1 + F'_L)·(1 + F'_{L-1})·..., keeping the product from collapsing to 0
        x_added_1 = X + context 
        x_norm_1 = self.norm1(x_added_1) # Normalize
        
        # Take each song's 64-dim vector, expand to 256. Each of the 256 outputs is a different weighted combination
        # of the 64 inputs
        # h[0] = w₀,₀·x[0] + w₀,₁·x[1] + ... + w₀,₆₃·x[63] + b[0]
        # h[1] = w₁,₀·x[0] + w₁,₁·x[1] + ... + w₁,₆₃·x[63] + b[1]
        # ...
        # h[255] = ...
        # Each h[i] is a learned feature detector. Say h[1] "lights up" when it detects "Slow and Minor Key Feel"
        # Results in 256 different lit up nodes representing 256 measurements
        ffn_out = self.ffn_expand(x_norm_1)

        # ReLU to strip out the negative outputs. We only care about detected features (positive output), negative
        # is just noise. Results in a vector that shows "the pattern is here to this extent (0 = not present)" for all 256 features per song. 
        # Here the network decides which patterns are present in each song
        ffn_out = self.ffn_relu(ffn_out)

        # Takes each song's 256-dim vector representing activations for 256 features and compresses back down to 64 features.
        # y[j] = v[j][0]·h[0] + v[j][1]·h[1] + ... + v[j][255]·h[255] + c[j]
        # Each out of the 64 outputs per song is a learned weighted combination of which patterns fired (h[x] per song)
        ffn_out = self.ffn_compress(ffn_out)
        
        # Add normalized X after context was added + FFN output (detected patterns)
        final_added = x_norm_1 + ffn_out
        out = self.norm2(final_added) # Normalize
        
        return out

# Loss Function, takes in model's raw output scores and the correct answers, and produces a single scalar measuring how wrong the model is
def cross_entropy_loss(logits, targets):
    """
    Combines Softmax activation and Cross-Entropy Loss into a single, 
    numerically stable, and mathematically optimized execution node.
    """
    targets_data = np.asarray(targets, dtype=np.float32)
    N = logits.data.shape[0]  # The batch size

    shifted_logits = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exps = np.exp(shifted_logits)
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    # Add a tiny epsilon (1e-15) to the probabilities to prevent np.log(0) crashes.
    # loss = -log(probability assigned to the correct answer by the model), averaged over the batch
    # targets_data is one-hot (1 at the correct song, 0 elsewhere), so multiplying zeroes out every
    # term EXCEPT the correct song's log-prob — that's how the sum reduces to -log(prob of correct answer).
    loss_val = -np.sum(targets_data * np.log(probs + 1e-15)) / N
    
    out = Tensor(loss_val, _creators=[logits], _op="CrossEntropy")
    
    def _backward():
        # Softmax + cross-entropy fused -> clean, easy to find gradient
        # "Predicted - True" This is the FIRST gradient that flows back into the network (out.grad is the seed = 1, since this is the loss)
        logits.grad += out.grad * (probs - targets_data) / N
        
    out._backward = _backward
    return out

# IDs -> vectors -> context-aware cectors -> scores over all songs
class SongRecommender(Module):
    """The complete end-to-end Recommendation Engine."""
    def __init__(self, vocab_size, embed_dim, context_length, num_layers=12):
        super().__init__()
        
        # Song embedding, vocab_size number of songs , each song is a vector w/ embed_dim features
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        # Position embedding, each slot holds a vector representing what it means to be at that position, irrespective of the actual song in that position (Song-agnostic)
        # It has context_length vectors (the amount of songs the model processes in one go)
        self.position_embedding = Embedding(num_embeddings=context_length, embedding_dim=embed_dim)

        # Create a list of num_layers Transformer blocks. Real models use 12 to 96 of these.
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(embed_dim))
            
        # The matchmaker (final linear layer) mapping embed_dim -> vocab_size
        # Takes the final, refined "vibe" vector, dot products it against each song in the catalog, produces one score per song measuring
        # "how well does this song match up against the 'vibe' vector?"
        self.matchmaker = LinearLayer(in_features=embed_dim, out_features=vocab_size)

    def forward(self, sequence_ids):
        # Look up the embedding for each song ID in sequence_ids and each position vector in self.position_embedding
        x = self.embedding(sequence_ids)
        pos_emb = self.position_embedding(np.arange(sequence_ids.shape[1]))

        # Add them together, so you get "what this song is + what it means for a song to be played at this position" for each song
        x = x + pos_emb
        
        # Pass the data through the stacked Transformer Blocks
        # In Brief: Each block takes in: x of shape (batch, seq, embed_dim) - the current song vectors
        # Does: (The two-sublayer refinement)
        # - Attention, each song gathers context (QKV) from the others (weighted blend of Values)
        # - Add + Norm, add that context back to the inputs and normalize
        # - FFN, push each song's vector independently through Expand -> ReLU -> Compress (detect patterns, rewrite the vector)
        # - Add + Norm, add the FFN output back, normalize
        # The last song is now contextualized with every previous song, represents the summary of everything heard so far, becomes our basis for prediction (used in matchmaker)
        for block in self.blocks:
            x = block(x)
        
        # x.data shape is (32, 3, 64) -> (Batch, Sequence, Embed). We want (32, 64).
        # This gives us the last song's vector, this is the next song prediction
        # NOTE: NOT a Tensor
        final_vibe_data = x.data[:, -1, :] 
        
        # Wrap it in a new Tensor and record the operation for Autograd
        final_vibe = Tensor(final_vibe_data, _creators=[x], _op="slice_last_step")
        
        def _backward_slice():
            # Create an empty gradient grid of the original 3D shape (32, 3, 64)
            grad_update = np.zeros_like(x.data)
            # Route the returning gradient ONLY back into the final vector since forward only took the last position
            grad_update[:, -1, :] = final_vibe.grad
            x.grad += grad_update
            
        final_vibe._backward = _backward_slice

        # Calculate dot products of final_vibe against the database to find the match
        logits = self.matchmaker(final_vibe)
        
        return logits

def load_spotify_data(data_folder_name="data", max_playlists=5000):
    """Parses the official Spotify JSON format into raw text playlists."""
    raw_playlists = []
    
    # Resolve the exact physical path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    secure_data_path = os.path.join(base_dir, data_folder_name)
    print(f"Checking physical disk at: {secure_data_path}")
    
    # Check if the folder physically exists
    if not os.path.exists(secure_data_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Windows cannot see the 'data' folder at all.")
        
    # Read EVERYTHING in the folder (Bypassing glob)
    all_files = os.listdir(secure_data_path)
    print(f"Found {len(all_files)} total items in the folder.")
    
    # Manually filter for JSON files
    json_files = [f for f in all_files if f.endswith('.json')]
    
    if not json_files:
        print(f"WHAT PYTHON SEES IN THE FOLDER: {all_files[:10]}")
        raise FileNotFoundError("The folder exists, but Python sees NO files ending in '.json'. (Is OneDrive hiding them?)")
        
    json_files.sort()  # deterministic order across runs
    print(f"Found {len(json_files)} JSON files. Loading until {max_playlists} playlists collected...")

    for file_name in json_files:
        file_path = os.path.join(secure_data_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for playlist in data['playlists']:
            tracks = [track['track_name'] for track in playlist['tracks']]
            if len(tracks) > 3:
                raw_playlists.append(tracks)

        if len(raw_playlists) >= max_playlists:
            break

    raw_playlists = raw_playlists[:max_playlists]
    print(f"Successfully loaded {len(raw_playlists)} valid playlists.")
    return raw_playlists

# Tokenize: Convert each song (a string/URI) into an integer ID. Assign each unique song a number to map song -> ID
# Slice: Cut each playlist into fixed-size training examples of the form (input window -> next song), a playlist is a sequence,
# slide a window across it and for each window input X is those songs, target Y is the song that comes right after
# raw playlists (song names)
#    │ tokenize: song → integer ID  (build vocab)
#    ▼
# tokenized playlists ([10, 11, 12, 13, 14])
#    │ slide a window of context_length
#    ▼
# X = [10,11,12]  Y = 13     ← each is one training example
#    │
#    ▼
# X feeds model.forward (embedding → blocks → last pos → matchmaker → logits)
# Y becomes the one-hot target in cross_entropy_loss
def tokenize_and_slice(raw_playlists, context_length=3, test_split=0.1):
    """
    Builds the vocabulary (song ↔ ID maps) over all data.
    Splits playlists into train/test sets.
    Tokenizes + slices each set into (X, Y) integer arrays.
    Returns the train/test arrays, vocab_size, and the ID→song map (for turning predictions back into song names later).
    """
    print("\n--- TOKENIZATION & SPLITTING ---")
    
    # Build Dictionary on ALL data to prevent KeyErrors, this is the Tokenize step
    track_to_id = {}
    id_to_track = {}
    current_id = 0
    for playlist in raw_playlists:
        for track in playlist:
            if track not in track_to_id:
                track_to_id[track] = current_id
                id_to_track[current_id] = track
                current_id += 1
                
    vocab_size = len(track_to_id) # Num. of unqiue songs/catalog size
    print(f"Vocabulary built! Found {vocab_size} unique songs.")
    
    # Shuffle and split the raw playlists into train/test at the playlist level
    # 90/10 split, train model on first 90% of playlists, test model on last 10% of playlists
    np.random.shuffle(raw_playlists)
    split_idx = int(len(raw_playlists) * (1 - test_split))
    train_raw = raw_playlists[:split_idx]
    test_raw = raw_playlists[split_idx:]
    
    # Helper function to slice a specific subset
    def process_subset(subset):
        X_data, Y_data = [], []
        for playlist in subset:
            tokenized = [track_to_id[track] for track in playlist]
            # For each position i, X_data gets tokenized[i: i+context_length] - a window of context_length consecutive songs (the INPUT)
            # Y_data gets tokenized [i + context_length] - the very next song after that window (the TARGET)
            for i in range(len(tokenized) - context_length):
                X_data.append(tokenized[i : i + context_length])
                Y_data.append(tokenized[i + context_length])
        # X_data is what flows through the embedding -> transformer, Y_data only appears as the one-hot target in cross_entropy_loss
        return np.array(X_data), np.array(Y_data)

    # Process the sets independently
    X_train, Y_train = process_subset(train_raw)
    X_test, Y_test = process_subset(test_raw)
    
    print(f"Train Sequences: {len(X_train)} | Test (Vault) Sequences: {len(X_test)}")
    # vocab_size = the num. of unique songs in the catalog, id_to_track maps song IDs to the actual song name
    return X_train, Y_train, X_test, Y_test, vocab_size, id_to_track

def save_model(model, filename="transformer_weights.npz"):
    """Extracts raw numpy arrays from the model's Tensors and saves to disk."""
    weights = [p.data for p in model.parameters()]
    np.savez(filename, *weights)
    print(f"\n[SYSTEM] Weights successfully saved to {filename}")

def load_model(model, filename="transformer_weights.npz"):
    """Overwrites the random initialization with saved weights if they exist."""
    if os.path.exists(filename):
        loaded = np.load(filename)
        params = model.parameters()
        for i, p in enumerate(params):
            p.data = loaded[f'arr_{i}']
        print(f"\n[SYSTEM] Previous weights loaded from {filename}. Resuming training...")
    else:
        print(f"\n[SYSTEM] No saved weights found at {filename}. Starting from scratch.")

import matplotlib.pyplot as plt

def train_transformer():
    print("\n--- TRANSFORMER INITIALIZATION ---")
    DATA_FOLDER = "data"
    CONTEXT_LENGTH = 10 # Input window size "Predict the 11th song from 10"
    EMBED_DIM = 64 # Feature-vector size for each song
    WEIGHT_FILE = "spotify_transformer_weights.npz" # Where we save/load trained weights
    
    raw_playlists = load_spotify_data(DATA_FOLDER, max_playlists=5000)
    
    # Notice the new X_test and Y_test variables
    X_train, Y_train, X_test, Y_test, vocab_size, id_to_track = tokenize_and_slice(raw_playlists, CONTEXT_LENGTH)
    
    model = SongRecommender(vocab_size=vocab_size, embed_dim=EMBED_DIM, context_length = CONTEXT_LENGTH, num_layers=2)
    optimizer = SGD(model.parameters(), lr=0.5)
    
    load_model(model, WEIGHT_FILE) # Loads saved weights if it exists
    
    # One-Hot Encode the Train targets, converts the target song IDs into one-hot vectors
    # The y-side prep that leads to loss calculation
    # Make a zero matrix (num_examples, vocab_size) — one row per example, one column per song.
    # For each example i, set the column at Y_train[i] (the correct song's ID) to 1.0.
    num_train_samples = len(Y_train)
    Y_train_onehot = np.zeros((num_train_samples, vocab_size), dtype=np.float32)
    for i in range(num_train_samples):
        Y_train_onehot[i, Y_train[i]] = 1.0

    # Grab a small static validation subset from the vault to keep the evaluation fast
    # Helps measure generalization (if the model is learning not just memorizing) during training using
    # a small 500 X_inputs with 500 corresponding Y_targets
    val_size = min(500, len(X_test))
    X_val = X_test[:val_size]
    Y_val_raw = Y_test[:val_size]
    Y_val_onehot = np.zeros((val_size, vocab_size), dtype=np.float32)
    for i in range(val_size):
        Y_val_onehot[i, Y_val_raw[i]] = 1.0

    print("\n--- TRAINING & VALIDATION ---")
    epochs = 30 # Num. of full passes over the training data to do
    batch_size = 32 # How many examples to process per gradient update
    
    # Logging Arrays to record loss of each epoch
    # Train_loss tells us the model is learning, val_loss tells us if its generalizing or memorizing
    train_loss_history = []
    val_loss_history = []

    # Everything is staged
    # data        → X_train, Y_train_onehot, X_val, Y_val_onehot, vocab_size, id_to_track
    # model       → SongRecommender (random or loaded weights)
    # optimizer   → SGD over all model parameters
    # config      → epochs=30, batch_size=32
    # logging     → train/val loss histories
    for epoch in range(epochs):
        # Each epoch uses the exact same training set (X_train)
        # Shuffle to make each epoch's batch different. Makes sure our model doesn't learn order-specific patterns
        indices = np.arange(num_train_samples)
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        Y_shuffled_onehot = Y_train_onehot[indices]
        
        epoch_loss = 0.0 # Accumulator for the total loss this epoch
        
        # TRAINING BATCHES
        for start_idx in range(0, num_train_samples, batch_size):
            # Slice data into mini-batches of 32
            end_idx = min(start_idx + batch_size, num_train_samples)
            X_batch = X_shuffled[start_idx:end_idx]
            Y_batch_onehot = Y_shuffled_onehot[start_idx:end_idx]
            
            # Forward pass, run the batch through the model
            # Embedding -> transformer blocks -> last position -> matchmaker -> logits
            logits = model(X_batch)
            # Calculate loss
            loss = cross_entropy_loss(logits, Y_batch_onehot)
            
            # Clear old gradients from last training run
            optimizer.zero_grad()

            # Backward pass
            # Topo sort autograd graph (start with loss, go all the way down to first Tensors -> run _backward in reverse, filling in .grad for every weight
            loss.backward()

            # Update the weights with the new gradients
            optimizer.step()
            
            # Accumulate the loss, weighted by the batch's actual size. Multiplying by batch size means each example contributes equally
            # even if the last batch is smaller, creates a true per-examples average not a per-batch one
            epoch_loss += loss.data * (end_idx - start_idx)
        
        # Divide the accumulated loss by the total num. of examples -> average training loss for this epoch. Record it for the learning curve
        avg_train_loss = epoch_loss / num_train_samples
        train_loss_history.append(avg_train_loss)
        
        # VALIDATION CHECK (Testing the Vault data)
        # We run the forward pass, but we skip .backward() so it cannot learn from this!
        val_logits = model(X_val)
        val_loss_node = cross_entropy_loss(val_logits, Y_val_onehot)
        val_loss_history.append(val_loss_node.data)
        
        # Compute validation accuracy on test data using NDCG@10
        # Take the logit the model outputted, find the correct song, and find num. of songs the model put above that song
        # Calculate rank, rank = 1 + num. of songs scored strictly higher than the correct song
        # Give a final score given that rank, score = 1/log2(1 + rank), so ideal score is 1/log2(2) = 1. 
        # Use a log function since log is a smoother curve (top 3 answers are given similar scores, then a gradual dropoff). Normalize (divide by 1, but this is done trivially)
        K = 10
        true_scores = val_logits.data[np.arange(val_size), Y_val_raw][:, None] # Grab the score the model gave the correct song, per example
        ranks = np.sum(val_logits.data > true_scores, axis=1) + 1 # Find how many score outscored the correct song, add 1 to get rank
        gains = 1.0 / np.log2(ranks + 1)
        gains[ranks > K] = 0.0 # Not in top-K, set to 0
        val_ndcg = np.mean(gains) # Average score across all validation examples (across all (X_train, Y_target) pairs)

        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_node.data:.4f} | NDCG@10: {val_ndcg:.4f}")
        
        save_model(model, WEIGHT_FILE)

    # Plots training loss vs. validation loss over epochs
    # Both decreasing together = healthy learning, Train ↓ but val ↑ → overfitting (memorizing NOT learning)
    print("\nTraining complete. Generating Loss Curve...")
    plt.style.use('dark_background') # Looks better for terminal-based developers
    plt.plot(train_loss_history, label="Training Loss (Memorization)", color='cyan')
    plt.plot(val_loss_history, label="Validation Loss (True Understanding)", color='magenta')
    plt.title("Transformer Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_transformer()