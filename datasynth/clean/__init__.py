"""Data-cleaning layer: frame cache (store) + deterministic ops (executor).

The clean *agent* (LLM) decides which ops to apply; this package executes them
deterministically on locally-cached sample frames. The source DB is never mutated.
"""
