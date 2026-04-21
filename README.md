# Star-Attention-Phase-2-Distributed-Softmax-Aggregation


# Star Attention, Phase 2: Distributed Softmax Aggregation

A from-scratch implementation of Phase 2 of the Star Attention algorithm,
written while learning how attention actually works under the hood.

Based on the paper:
**"Star Attention: Efficient LLM Inference over Long Sequences"** by Acharya et al., 2025.

## What is this about?

When you feed a really long sequence into a Large Language Model (think
hundreds of thousands of tokens), the attention computation becomes the
bottleneck. The standard approach needs to hold the entire Key/Value cache
in one place, which gets expensive fast.

Star Attention splits this work across multiple machines (called "hosts").
Each host stores and processes only a chunk of the KV cache. The clever part
is Phase 2: how do you combine partial attention results from different hosts
into the exact answer you would have gotten from doing it all at once?

The answer involves a trick called the **online softmax** (or log-sum-exp trick),
which lets each host work with its own local max value for numerical stability,
while still allowing the query host to combine everything correctly.

## How it works (intuition)

Softmax needs to divide by a sum of exponentials. When you take `exp(big number)`
it overflows, so the standard fix is to subtract the max value first. But in a
distributed setting, each host only sees its own chunk, so it doesn't know the
global max.

The trick: let each host use its OWN local max, and send back enough info so
the query host can rescale everything onto a common scale before combining.

Each host sends just three things per query token:
1. `m_h`: its local max score
2. `s_h`: sum of exponentials (shifted by its local max)
3. `A_h`: weighted sum of values (also shifted)

The query host finds the true global max, rescales each host's contribution
using basic exponent rules (`exp(a) = exp(a - b) * exp(b)`), and combines
them into the final attention output.

## Running it

```bash
python star_attention_phase2.py
```

You only need numpy installed:

```bash
pip install numpy
```

## What the output looks like

```
============================================================
Star Attention, Phase 2: Distributed Softmax Aggregation
============================================================
  Hosts=4  QueryTokens=3  KV per host=16  Dim=8

  Host 1: local_max[token0]=1.243  s_h[token0]=5.1998
  Host 2: local_max[token0]=1.470  s_h[token0]=4.8802
  Host 3: local_max[token0]=1.972  s_h[token0]=3.7005
  Host 4: local_max[token0]=2.110  s_h[token0]=2.9460

============================================================
Checking against naive global attention
============================================================
  Max absolute error : 1.25e-16
  Exact match        : True

Communication per transformer layer:
  Naive (ship all KV) :   1024 floats
  Star Attention      :    120 floats  (8.5x reduction)
```

The max error is around `1e-16`, which is just floating point noise. The
distributed answer is mathematically exact, not an approximation.

## Why it matters

The big win is communication cost. In a naive setup, every host would need
to ship its entire KV cache to one place, which is `H * lk * d * 2` floats.
Star Attention only sends a tiny summary per query token, which is
`H * lq * (2 + d)` floats. For long sequences this is a huge saving, and it
gets better as sequences get longer.

## File structure

```
.
├── star_attention_phase2.py   # the main implementation
└── README.md                  # this file
```

## What I learned

- How the softmax stability trick actually works and why it's needed
- That you can combine softmax results across partitions exactly (not just
  approximately) if you keep track of the local maxes
- Why attention is expensive to parallelize naively, and how clever math
  can reduce communication cost by an order of magnitude
- How exponent rules (`exp(a) = exp(a - b) * exp(b)`) power a lot of
  numerical tricks in deep learning

## References

- Acharya et al., "Star Attention: Efficient LLM Inference over Long Sequences", 2025
- The online softmax trick comes from earlier work on memory-efficient
  attention (FlashAttention and related)
