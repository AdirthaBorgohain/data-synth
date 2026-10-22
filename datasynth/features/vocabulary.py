"""Fixed feature-spec vocabularies.

The plan/critique LLM picks ONLY from these (like the clean phase's OP_VOCABULARY),
so every spec compiles to deterministic, injection-safe SQL or polars. Descriptions
are surfaced to the model in the node prompts.
"""

from __future__ import annotations

# SQL aggregate functions allowed on child tables (mapped to SQL in sql_build).
# "count" ignores source_column; "recency_days" needs a date source_column.
AGG_FUNCS: dict[str, str] = {
    "count": "number of child rows per entity (source_column ignored)",
    "count_distinct": "distinct values of source_column per entity",
    "sum": "sum of a numeric source_column per entity",
    "avg": "mean of a numeric source_column per entity",
    "min": "minimum of source_column per entity",
    "max": "maximum of source_column per entity",
    "stddev": "standard deviation of a numeric source_column per entity",
    "recency_days": "days since the most recent date source_column for the entity",
}

# In-memory polars transforms applied to the assembled matrix (after SQL).
TRANSFORM_VOCABULARY: dict[str, str] = {
    "log": "log1p of a non-negative numeric feature (params: input)",
    "ratio": "input_a / input_b, div-by-zero → null (params: input_a, input_b)",
    "interaction": "product input_a * input_b (params: input_a, input_b)",
    "bin": "quantile-bin a numeric feature into k buckets (params: input, k)",
    "datetime_part": "extract year/month/dow from a date feature (params: input, part)",
    "tenure_days": "days between a date feature and now (params: input)",
    "missing_indicator": "1 where a feature is null else 0 (params: input)",
}

# Categorical encoders applied to the assembled matrix.
ENCODE_VOCABULARY: dict[str, str] = {
    "frequency": "replace each category with its frequency count (params: column)",
    "target": "replace each category with the mean target for that category (params: column)",
    "one_hot": "one-hot indicator columns for a low-cardinality category (params: column)",
}
