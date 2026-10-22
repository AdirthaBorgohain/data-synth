"""Finding-scoped plot rendering.

A finding cites the (table, analysis) it draws on; this module materialises a
single annotated PNG per such finding, using the already-computed
`eda_results[table][analysis]` dict and (when needed) the cached sample frame
from the FrameStore. The PNG is captioned with the finding's topic, observation,
and downstream implication so it stands alone outside the CLI.

Non-plottable analyses (numeric_summary, cardinality) and analyses with thin
results (e.g. missingness with no high-null columns) return None so the caller
prints the finding text without a plot link.
"""

from __future__ import annotations

from datasynth.artifacts.store import ArtifactStore, slugify
from datasynth.clean.store import FrameStore
from datasynth.eda import plots


def _slug(topic: str, idx: int) -> str:
    return f"finding_{idx:02d}__{slugify(topic)}"


def _save_args(finding: dict) -> dict:
    # PNG carries the analysis's axis title plus the observation as a wrapped caption.
    return {"caption": finding.get("observation") or None}


def _render_correlation(result, frame, artifacts, slug, finding):
    matrix = result.get("matrix")
    labels = result.get("columns")
    if not matrix or not labels or len(labels) < 2:
        return None
    return plots.correlation_heatmap(matrix, labels, artifacts, slug, **_save_args(finding))


def _render_distributions(result, frame, artifacts, slug, finding):
    # Pick the most skewed numeric column from the frame (best signal in a single histogram).
    if frame is None:
        return None
    candidates = [c for c in result.get("numeric_columns", []) if c in frame.columns]
    if not candidates:
        return None
    target_col = None
    best_skew = -1.0
    for c in candidates:
        try:
            sk = abs(frame[c].drop_nulls().skew() or 0.0)
        except Exception:
            sk = 0.0
        if sk > best_skew:
            best_skew = sk
            target_col = c
    if target_col is None:
        target_col = candidates[0]
    return plots.histogram(
        frame[target_col].to_list(),
        f"Distribution of {target_col}",
        artifacts, slug, **_save_args(finding),
    )


def _render_categorical_freq(result, frame, artifacts, slug, finding):
    # Pick the categorical column with the largest top-value share (most informative).
    if not result:
        return None
    best_col, best_share, best_entries = None, -1.0, None
    for col, entries in result.items():
        if not entries:
            continue
        total = sum(e["count"] for e in entries) or 1
        share = entries[0]["count"] / total
        if share > best_share:
            best_col, best_share, best_entries = col, share, entries
    if not best_col:
        return None
    labels = [e["value"] for e in best_entries]
    counts = [e["count"] for e in best_entries]
    return plots.bar_counts(
        labels, counts, f"Top values — {best_col}", artifacts, slug, **_save_args(finding),
    )


def _render_outliers(result, frame, artifacts, slug, finding):
    if frame is None:
        return None
    flagged = [(c, v) for c, v in result.items()
               if isinstance(v, dict) and v.get("n_outliers", 0) > 0 and c in frame.columns]
    if not flagged:
        return None
    worst_col = max(flagged, key=lambda kv: kv[1]["pct_outliers"])[0]
    return plots.boxplot(
        frame[worst_col].to_list(),
        f"Outliers — {worst_col}", artifacts, slug, **_save_args(finding),
    )


def _render_missingness(result, frame, artifacts, slug, finding):
    rates = result.get("null_rates") or {}
    if not rates:
        return None
    columns = list(rates.keys())
    values = [rates[c] for c in columns]
    return plots.missingness_bar(columns, values, artifacts, slug, **_save_args(finding))


def _render_datetime_trends(result, frame, artifacts, slug, finding):
    # Use the date column with the longest trend series.
    best = None
    for col, info in (result or {}).items():
        if not isinstance(info, dict):
            continue
        months, counts = info.get("months"), info.get("counts")
        if months and counts and (best is None or len(months) > len(best[1])):
            best = (col, months, counts)
    if not best:
        return None
    col, months, counts = best
    return plots.line_counts(
        months, counts, f"Row count over time — {col}", "month",
        artifacts, slug, **_save_args(finding),
    )


def _render_target_analysis(result, frame, artifacts, slug, finding):
    task = result.get("task_type")
    target = result.get("target")
    if task == "classification":
        balance = result.get("class_balance") or []
        if not balance:
            return None
        labels = [b["class"] for b in balance]
        counts = [b["count"] for b in balance]
        return plots.bar_counts(
            labels, counts, f"Class balance — {target}", artifacts, slug, **_save_args(finding),
        )
    if task == "regression" and frame is not None and target in frame.columns:
        return plots.histogram(
            frame[target].to_list(), f"Target distribution — {target}",
            artifacts, slug, **_save_args(finding),
        )
    return None


_DISPATCH = {
    "correlation": _render_correlation,
    "distributions": _render_distributions,
    "categorical_freq": _render_categorical_freq,
    "outliers": _render_outliers,
    "missingness": _render_missingness,
    "datetime_trends": _render_datetime_trends,
    "target_analysis": _render_target_analysis,
    # numeric_summary + cardinality are intentionally non-plottable.
}


def render_finding(
    finding: dict,
    eda_results: dict,
    frame_store: FrameStore,
    artifacts: ArtifactStore,
    idx: int,
) -> str | None:
    """Render a single annotated PNG for a finding; return its path or None."""
    table = finding.get("table") or ""
    analysis = finding.get("analysis") or ""
    fn = _DISPATCH.get(analysis)
    if fn is None:
        return None
    result = (eda_results.get(table) or {}).get(analysis)
    if not isinstance(result, dict) or "error" in result:
        return None
    frame = frame_store.get(table) if frame_store.has(table) else None
    try:
        return fn(result, frame, artifacts, _slug(finding.get("topic", ""), idx), finding)
    except Exception:  # noqa: BLE001
        return None
