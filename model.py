import numpy as np
from engine import Tensor, Module

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
        
        # Scale down scores, wrap as Tensor and give a backward so backprop can travel through and account for the scaling
        scaled_data = scores.data / self.scale
        scaled = Tensor(scaled_data, _creators=[scores], _op="scale")

        def scale_backward():
            scores.grad += scaled.grad / self.scale
        scaled._backward = scale_backward

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
# Measures how much probability the model put on the correct answer, the lower the probability to the true song the higher the loss
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