"""
Star Attention - Phase 2: Distributed Softmax Aggregation

So I was reading this paper called "Star Attention: Efficient LLM Inference
over Long Sequences" (Acharya et al., 2025) and tried implementing Phase 2
myself to actually understand how it works.

The big idea: when you have a really long sequence, you can split the
key/value cache across multiple machines (hosts). Each host computes attention
over its own chunk, and then we combine the results on one "query host".

The tricky part is the softmax. Softmax needs to know the MAX value across
ALL scores to be numerically stable (otherwise exp() blows up). But each host
only sees its own chunk, so it doesn't know the global max.

The trick is something called the "online softmax" or "log-sum-exp" trick.
Each host uses its OWN local max for stability, sends some extra info, and
the query host can mathematically combine everything to get the exact same
answer as if we had done it all in one place.

Let me walk through it step by step below.
"""

import numpy as np

# Setting a seed so I get the same random numbers every time I run this
np.random.seed(42)

# Configuration
# I'm using small numbers here so it's easy to reason about what's happening
H = 4    # number of hosts (machines) we're splitting the work across
lq = 3   # number of query tokens (the things asking "what should I attend to?")
lk = 16  # number of key/value tokens stored on EACH host
d  = 8   # dimension of each attention head (vector size)

print("=" * 60)
print("Star Attention, Phase 2: Distributed Softmax Aggregation")
print("=" * 60)
print(f"  Hosts={H}  QueryTokens={lq}  KV per host={lk}  Dim={d}")
print()

# Q is our query matrix. Think of each row as a token "asking a question"
Q = np.random.randn(lq, d)

# Each host has its own slice of Keys and Values
# In a real system these would live on different machines
Ks = [np.random.randn(lk, d) for _ in range(H)]
Vs = [np.random.randn(lk, d) for _ in range(H)]

# STEP 1: Each host does its own local computation
#
# Each host needs to send 3 things back to the query host:
#   m_h : the maximum score it saw (one number per query token)
#   s_h : sum of exp(scores), shifted by its local max (for stability)
#   A_h : sum of exp(scores) times values, also shifted (the "numerator")
#
# Why send the max? Because the query host needs to know how each host
# shifted its numbers so it can "un-shift" them and combine correctly.

local_m = []  # will store each host's local maxes
local_s = []  # will store each host's local sums
local_A = []  # will store each host's local weighted value sums

for h in range(H):
    # Standard scaled dot-product attention scores: Q @ K^T / sqrt(d)
    # The sqrt(d) keeps the scores from getting too large when d is big
    scores = Q @ Ks[h].T / np.sqrt(d)          # shape: (lq, lk)

    # Find the largest score for each query token (within this host's chunk)
    m_h = scores.max(axis=1)                   # shape: (lq,)

    # Subtract the max BEFORE exponentiating
    # This is the classic numerical stability trick: exp(big number) overflows,
    # but exp(score minus max) is always between 0 and 1
    e = np.exp(scores - m_h[:, None])          # shape: (lq, lk)

    # Sum of the exponentials (this would be the denominator of softmax)
    s_h = e.sum(axis=1)                        # shape: (lq,)

    # Weighted sum of values (this would be the numerator of attention output)
    # Notice we're NOT dividing by s_h yet, because we can't normalize
    # until we've combined results from all hosts
    A_h = e @ Vs[h]                            # shape: (lq, d)

    local_m.append(m_h)
    local_s.append(s_h)
    local_A.append(A_h)
    print(f"  Host {h+1}: local_max[token0]={m_h[0]:.3f}  s_h[token0]={s_h[0]:.4f}")

# STEP 2: The query host combines everything (the magic part!)
#
# Each host shifted its numbers by a DIFFERENT local max.
# To combine them correctly, we need to put everyone on the same scale.
#
# Math intuition: if host h shifted by m_h, then to convert its numbers
# to the "global max" scale, we multiply by exp(m_h minus global_max).
# This is just basic exponent rules: exp(a) = exp(a-b) * exp(b)

# First, find the true global max by taking the max across all hosts' maxes
global_max = np.stack(local_m, axis=0).max(axis=0)   # shape: (lq,)

# Now combine all the local sums, rescaling each one to the global max scale
s_global = sum(
    local_s[h] * np.exp(local_m[h] - global_max)
    for h in range(H)
)   # shape: (lq,)

# Same idea for the value-weighted sums, then divide by the global denominator
# This last division is the actual softmax normalization
A_global = sum(
    local_A[h] * np.exp(local_m[h] - global_max)[:, None]
    for h in range(H)
) / s_global[:, None]   # shape: (lq, d)

# STEP 3: Sanity check by computing attention the "normal" way
#
# Just concatenate all the keys and values together and do regular attention.
# If our distributed version is correct, both should give the exact same answer.

K_all = np.concatenate(Ks, axis=0)             # shape: (H*lk, d)
V_all = np.concatenate(Vs, axis=0)
sc = Q @ K_all.T / np.sqrt(d)
sc -= sc.max(axis=1, keepdims=True)            # numerical stability
e_all = np.exp(sc)
A_naive = (e_all / e_all.sum(axis=1, keepdims=True)) @ V_all

# Compare the two approaches
max_err = np.abs(A_global - A_naive).max()
match = np.allclose(A_global, A_naive, atol=1e-9)

print()
print("=" * 60)
print("Checking against naive global attention")
print("=" * 60)
print(f"  Max absolute error : {max_err:.2e}")
print(f"  Exact match        : {match}")
print()

# Why does any of this matter? Communication cost.
#
# In a naive approach, we'd have to ship ALL the keys and values to one place.
# With Star Attention, each host only sends a tiny summary back.
# For long sequences this is a HUGE saving.

per_host = lq * (1 + 1 + d)        # m_h (1) + s_h (1) + A_h (d) per query token
kv_naive = H * lk * d * 2          # all keys and all values from all hosts
star_comm = H * per_host           # what star attention actually sends

print("Communication per transformer layer:")
print(f"  Naive (ship all KV) : {kv_naive:>6} floats")
print(f"  Star Attention      : {star_comm:>6} floats  ({kv_naive/star_comm:.1f}x reduction)")
