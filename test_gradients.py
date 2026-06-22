import numpy as np
import song_recommender
from song_recommender import SongRecommender, cross_entropy_loss

# Checks if we are using the right math to find gradients. Compares backprop-generated gradients w/ manually derived gradients
# (found using slope)
def gradient_check(eps=1e-3, tol=1e-2, samples_per_param=8, seed=0):
    song_recommender.DEFAULT_DTYPE = np.float64 # gradient check in double precision
    np.random.seed(seed)  # model init uses the global RNG -> reproducible runs

    # Initialize tiny model + tiny batch
    # Every entry we check costs two forward passes (w + e and w - e). Keep everything small
    vocab_size, embed_dim, context_length, num_layers, batch = 20, 8, 4, 1, 3
    model = SongRecommender(vocab_size, embed_dim, context_length, num_layers=num_layers)

    X = np.random.randint(0, vocab_size, size=(batch, context_length))
    Y = np.random.randint(0, vocab_size, size=(batch,))
    Y_onehot = np.zeros((batch, vocab_size), dtype=np.float32)
    Y_onehot[np.arange(batch), Y] = 1.0

    def loss_value(): # one forward pass -> scalar loss
        return cross_entropy_loss(model(X), Y_onehot).data
    
    # Find Analytical Gradients (using built-in backprop)
    # One forward + backward pass fills .grad on every parameter. Save copies now, because the numerical step 
    # below only does forward passes (never backward)
    params = model.parameters()
    for p in params:
        p.grad = np.zeros_like(p.data)
    
    cross_entropy_loss(model(X), Y_onehot).backward()
    analytical = [p.grad.copy() for p in params]

    # Find Numerical Gradients (manual, use slope b/n "w - e" and "w + e" where w is a weight and e is a small change)
    rng = np.random.RandomState(seed + 1)
    max_rel = 0.0
    worst = (0.0, None, 0.0, 0.0)  # rel, op, ana, numerical
    print(f"{'idx':>3}  {'op':<12} {'shape':<14} {'max_rel':>9}")
    for pi, p in enumerate(params):
        ana_flat = analytical[pi].reshape(-1)
        # sample a few entries instead of checking every one (speed)
        idxs = rng.choice(p.data.size, size=min(samples_per_param, p.data.size), replace=False)
        param_max = 0.0
        for idx in idxs:
            orig = p.data.flat[idx]
            p.data.flat[idx] = orig + eps # nudge UP
            l_plus = loss_value()
            p.data.flat[idx] = orig - eps # nudge DOWN
            l_minus = loss_value()
            p.data.flat[idx] = orig # restore EXACTLY

            numerical = (l_plus - l_minus) / (2 * eps) # the two-sided slope (w - e and w + e)
            ana = ana_flat[idx]
    
            # Compare Analytical Gradient and Numerical Gradients. Since analytical calculates instantaneous slope (derivative)
            # and numerical calculates slope (not instantaneous), they should be very close to each other if not equal
            # Edge case: unused embedding rows have grad 0 both ways 0/0 is meaningless -> skip them.
            if abs(ana) < 1e-7 and abs(numerical) < 1e-7:
                continue
            rel = abs(ana - numerical) / max(abs(ana) + abs(numerical), 1e-8)
            param_max = max(param_max, rel)
            if rel > worst[0]:
                worst = (rel, p._op, ana, numerical)
        max_rel = max(max_rel, param_max)
        flag = "  <-- FAIL" if param_max > tol else ""
        print(f"{pi:>3}  {str(p._op):<12} {str(p.data.shape):<14} {param_max:>9.2e}{flag}")

    
    print(f"\nWorst entry: op={worst[1]}  analytical={worst[2]:.4e}  numerical={worst[3]:.4e}")
    print(f"Max relative error: {max_rel:.2e}  (threshold {tol:.0e})")
    print("PASS" if max_rel < tol else "FAIL")
    return max_rel

if __name__ == "__main__":
    gradient_check()