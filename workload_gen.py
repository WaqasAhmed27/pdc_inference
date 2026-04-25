"""
PDC Workload Generator — Phase 2
Generates synthetic editor-style prompts that simulate real-world usage:
  - Cursor-driven autocomplete (short completions mid-sentence)
  - Short rewrite / grammar-correction tasks
  - Wikipedia-revision-style incremental edits (varying context lengths)

All workloads return lists of dicts compatible with the /complete and /rewrite endpoints.
"""

import random
from typing import List

# ---------------------------------------------------------------------------
# Seed corpora
# ---------------------------------------------------------------------------

# NLP / systems topics — realistic autocomplete mid-sentence seeds
_AUTOCOMPLETE_SEEDS = [
    "The transformer architecture was introduced in 2017 and has since",
    "Memory efficiency in large language models is critical because",
    "The KV cache stores key and value tensors from previous tokens, allowing",
    "Speculative decoding works by using a smaller draft model to",
    "When a user types in an AI-assisted editor, the system must",
    "Parallel computing techniques can significantly reduce the time required to",
    "The attention mechanism computes a weighted sum of values based on",
    "GPU memory bandwidth is often the limiting factor in",
    "Quantization reduces model size by representing weights with fewer bits, which",
    "Inference latency is affected by both the prefill phase and",
    "The time-to-first-token metric measures how long it takes before",
    "Distributed training splits computation across multiple devices to",
    "Cache eviction policies determine which entries to remove when",
    "Real-time text editing requires the model to respond within",
    "Modern CPUs include multiple cores that can execute tasks",
    "The NUMA architecture affects memory access latency because",
    "Thread affinity pins threads to specific CPU cores, which",
    "Throughput and latency are often in tension: optimizing for",
    "The prefill phase processes the entire input prompt in parallel, while",
    "Continuous batching allows the server to add new requests to",
    # Wikipedia-style factual fragments
    "The Battle of Waterloo was fought on 18 June 1815, marking",
    "Photosynthesis is the process by which plants convert light energy into",
    "The human genome contains approximately 3 billion base pairs and",
    "Climate change refers to long-term shifts in global temperatures and",
    "The French Revolution began in 1789 and fundamentally changed",
    "Neural networks consist of layers of interconnected nodes that",
    "The Python programming language was first released in 1991 and",
    "Operating systems manage hardware resources by providing",
    "The internet protocol suite, commonly known as TCP/IP,",
    "Reinforcement learning is a type of machine learning in which",
]

# Informal / poorly-written sentences for rewrite tasks
_REWRITE_SEEDS = [
    "The thing is that transformers work by doing attention stuff on tokens which lets them understand context.",
    "GPU memory can run out if you have too many things in the KV cache which is bad for performance.",
    "Speculative decoding is when you use a small model to guess tokens and then check them with the big model.",
    "The latency problem in text editors is that users have to wait too long which makes the experience bad.",
    "Quantization makes models smaller and faster but sometimes the quality gets worse depending on how much you quantize.",
    "Parallel computing is useful because you can split work across many processors which makes it go faster.",
    "The benchmark measures how fast the system is by sending requests and seeing how long they take to finish.",
    "Memory efficiency matters a lot for running models on regular computers that don't have lots of VRAM.",
    "The evaluation framework tests the system under different conditions to see how it performs in each case.",
    "Thread pinning helps performance because cache misses are expensive when threads move between cores.",
    "The model loads slower at first because it has to read all the weights from disk into memory.",
    "Batching requests together can make the GPU work more efficiently but it also adds some delay.",
    "The KV cache gets bigger as the context window grows which can cause out-of-memory errors.",
    "Draft models need to be fast otherwise speculative decoding doesn't actually help with latency.",
    "You can measure throughput by counting how many tokens per second the system can generate total.",
]

_REWRITE_INSTRUCTIONS = [
    "Improve clarity and conciseness.",
    "Rewrite in a formal academic tone.",
    "Make this more precise and technical.",
    "Simplify this for a general audience.",
    "Correct any grammatical issues and improve flow.",
]

# A realistic document fragment for revision-history simulation
_BASE_DOC = (
    "The transformer model architecture, introduced by Vaswani et al. in 2017, "
    "revolutionized natural language processing by replacing recurrent networks with "
    "a self-attention mechanism. This allows the model to process all tokens in parallel "
    "during training, dramatically reducing training time compared to LSTMs and GRUs. "
    "The core innovation is the multi-head attention mechanism, which computes attention "
    "weights between all token pairs simultaneously. Each attention head learns a different "
    "aspect of the relationships between tokens, and their outputs are concatenated and "
    "projected before being passed to a feed-forward network. Layer normalisation and "
    "residual connections stabilise training and allow for very deep architectures. "
    "The model uses positional encodings to inject information about token order, since "
    "the attention mechanism itself is permutation-invariant. This architecture forms the "
    "backbone of modern language models including BERT, GPT, and their successors."
)

# ---------------------------------------------------------------------------
# Workload generators
# ---------------------------------------------------------------------------

def generate_autocomplete_workload(n: int = 20, seed: int = 42) -> List[dict]:
    """
    Returns n autocomplete payloads for /complete.
    max_tokens is varied to simulate short (32) and medium (64) completions.
    """
    random.seed(seed)
    pool = _AUTOCOMPLETE_SEEDS * ((n // len(_AUTOCOMPLETE_SEEDS)) + 2)
    random.shuffle(pool)
    return [
        {
            "prompt": prompt,
            "max_tokens": random.choice([32, 48, 64]),
            "temperature": 0.2,
            "stream": True,
        }
        for prompt in pool[:n]
    ]


def generate_rewrite_workload(n: int = 20, seed: int = 42) -> List[dict]:
    """
    Returns n rewrite payloads for /rewrite.
    """
    random.seed(seed)
    pool = _REWRITE_SEEDS * ((n // len(_REWRITE_SEEDS)) + 2)
    random.shuffle(pool)
    return [
        {
            "text": text,
            "instruction": random.choice(_REWRITE_INSTRUCTIONS),
            "max_tokens": 128,
            "temperature": 0.3,
            "stream": True,
        }
        for text in pool[:n]
    ]


def generate_revision_history_workload(n: int = 20, seed: int = 42) -> List[dict]:
    """
    Simulates cursor-driven editing by slicing a base document at increasing offsets.
    This mimics the KV-cache pattern seen during real incremental typing:
    each request shares a long prefix with the previous one.

    Shorter slices = cold-start / short-context requests.
    Longer slices = warm / long-context requests.
    """
    random.seed(seed)
    doc_len = len(_BASE_DOC)
    # Evenly spaced cut points across the document, then shuffle
    cuts = [int(doc_len * i / n) for i in range(1, n + 1)]
    random.shuffle(cuts)

    return [
        {
            "prompt": _BASE_DOC[:cut],
            "max_tokens": 32,
            "temperature": 0.1,
            "stream": True,
        }
        for cut in cuts[:n]
    ]


def generate_mixed_workload(n: int = 30, seed: int = 42) -> List[dict]:
    """
    Mix of autocomplete and rewrite tasks, randomly interleaved.
    Useful for concurrent-session stress tests.
    Returns (endpoint, payload) tuples.
    """
    random.seed(seed)
    ac = generate_autocomplete_workload(n, seed)
    rw = generate_rewrite_workload(n, seed)

    mixed = [("complete", p) for p in ac] + [("rewrite", p) for p in rw]
    random.shuffle(mixed)
    return mixed[:n]


# ---------------------------------------------------------------------------
# CLI preview
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Autocomplete sample ===")
    for item in generate_autocomplete_workload(3):
        print(f"  [{item['max_tokens']} tok] {item['prompt'][:70]}...")

    print("\n=== Rewrite sample ===")
    for item in generate_rewrite_workload(3):
        print(f"  [{item['instruction']}]\n  {item['text'][:70]}...")

    print("\n=== Revision history sample ===")
    for item in generate_revision_history_workload(4):
        print(f"  [ctx={len(item['prompt'])} chars] {item['prompt'][-50:]!r}...")
