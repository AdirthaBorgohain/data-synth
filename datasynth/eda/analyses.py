"""Deterministic EDA analyses over polars frames.

The propose-LLM picks analyses from EDA_VOCABULARY (a fixed menu, like the clean
phase's OP_VOCABULARY); run_analyses executes them on the cached sample frames and
returns JSON-serializable result dicts. Plots are deferred — eda_render reads
these results plus the cached frame and produces only the PNGs that surfaced
findings cite. Individual analysis failures are recorded, never raised, so one
bad analysis can't abort EDA.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from datasynth._dtypes import CLASS_MAX_UNIQUE as _CLASS_MAX_UNIQUE
from datasynth._dtypes import NUMERIC as _NUMERIC
from datasynth._dtypes import TEMPORAL as _TEMPORAL
from datasynth.artifacts.store import ArtifactStore

# analysis name → description, shown to the propose-LLM.
EDA_VOCABULARY: dict[str, str] = {
    "numeric_summary": "descriptive stats per numeric column (mean/std/quartiles/skew/kurtosis)",
    "distributions": "histogram of each numeric column's distribution",
    "categorical_freq": "top value frequencies for categorical columns (bar chart)",
    "correlation": "pairwise correlation matrix of numeric columns (heatmap) + strong pairs",
    "outliers": "IQR-based outlier counts per numeric column (boxplot)",
    "missingness": "null-rate summary across columns (bar chart)",
    "cardinality": "constant, near-unique (likely IDs), and high-cardinality columns",
    "datetime_trends": "row-count trend over time for date/datetime columns",
    "target_analysis": "target class balance / distribution and target-vs-feature breakdown",
}

# Columns with unique-ratio at or above this are flagged as likely identifiers.
_ID_UNIQUE_RATIO = 0.95


def _clean(x: Any) -> Any:
    """Make a scalar JSON-safe (NaN/inf → None, numpy/polars → python)."""
    if x is None:
        return None
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
    except (TypeError, ValueError):
        return x
    return x


def _numeric_cols(df: pl.DataFrame, only: list[str] | None = None) -> list[str]:
    cols = only or df.columns
    return [c for c in cols if c in df.columns and df[c].dtype in _NUMERIC]


def _temporal_cols(df: pl.DataFrame, only: list[str] | None = None) -> list[str]:
    cols = only or df.columns
    return [c for c in cols if c in df.columns and df[c].dtype in _TEMPORAL]


def _categorical_cols(df: pl.DataFrame, only: list[str] | None = None) -> list[str]:
    cols = only or df.columns
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        dt = df[c].dtype
        if dt not in _NUMERIC and dt not in _TEMPORAL:
            out.append(c)
    return out


# ── individual analyses ──────────────────────────────────────────────────────
# Each returns (result_dict, plot_meta_list). plot_meta = {title, path, kind}.

def _numeric_summary(df, cols, params, target, artifacts, cfg):
    result: dict[str, dict] = {}
    for c in _numeric_cols(df, cols):
        s = df[c].drop_nulls()
        if s.len() == 0:
            continue
        result[c] = {
            "count": int(s.len()),
            "mean": _clean(s.mean()),
            "std": _clean(s.std()),
            "min": _clean(s.min()),
            "p25": _clean(s.quantile(0.25)),
            "median": _clean(s.median()),
            "p75": _clean(s.quantile(0.75)),
            "max": _clean(s.max()),
            "skew": _clean(s.skew()),
            "kurtosis": _clean(s.kurtosis()),
        }
    return result, []


def _distributions(df, cols, params, target, artifacts, cfg):
    # Plots are deferred to eda_render (per-finding); analyses compute only.
    return {"numeric_columns": _numeric_cols(df, cols)}, []


def _categorical_freq(df, cols, params, target, artifacts, cfg):
    top_k = int(cfg.get("top_k", 10))
    result: dict[str, list] = {}
    for c in _categorical_cols(df, cols):
        vc = df[c].drop_nulls().value_counts(sort=True).head(top_k)
        if vc.height == 0:
            continue
        count_col = "count" if "count" in vc.columns else vc.columns[-1]
        labels = vc[c].to_list()
        counts = [int(x) for x in vc[count_col].to_list()]
        result[c] = [{"value": str(l), "count": n} for l, n in zip(labels, counts)]
    return result, []


def _correlation(df, cols, params, target, artifacts, cfg):
    num = _numeric_cols(df, cols)
    if len(num) < 2:
        return {"note": "fewer than 2 numeric columns; correlation skipped"}, []
    corr = df.select(num).corr()
    matrix = [[_clean(v) for v in corr[c].to_list()] for c in corr.columns]
    threshold = float(cfg.get("corr_threshold", 0.7))
    strong = []
    for i in range(len(num)):
        for j in range(i + 1, len(num)):
            r = matrix[i][j]
            if r is not None and abs(r) >= threshold:
                strong.append({"a": num[i], "b": num[j], "r": round(r, 3)})
    return {"columns": num, "matrix": matrix, "strong_pairs": strong,
            "threshold": threshold}, []


def _outliers(df, cols, params, target, artifacts, cfg):
    result: dict[str, dict] = {}
    for c in _numeric_cols(df, cols):
        s = df[c].drop_nulls()
        if s.len() == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        if q1 is None or q3 is None:
            continue
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(s.filter((s < lo) | (s > hi)).len())
        result[c] = {
            "lower_fence": _clean(lo), "upper_fence": _clean(hi),
            "n_outliers": n_out, "pct_outliers": round(n_out / s.len(), 4),
        }
    return result, []


def _missingness(df, cols, params, target, artifacts, cfg):
    n = df.height
    result: dict[str, float] = {}
    for c in df.columns:
        rate = round(df[c].null_count() / n, 4) if n else 0.0
        result[c] = rate
    return {"null_rates": result}, []


def _cardinality(df, cols, params, target, artifacts, cfg):
    n = df.height
    constant, likely_ids, high_card = [], [], []
    for c in df.columns:
        nu = df[c].n_unique()
        if nu <= 1:
            constant.append(c)
            continue
        ratio = nu / n if n else 0.0
        if ratio >= _ID_UNIQUE_RATIO and df[c].dtype not in _NUMERIC:
            likely_ids.append(c)
        elif df[c].dtype not in _NUMERIC and nu > int(cfg.get("top_k", 10)) * 5:
            high_card.append({"column": c, "n_unique": int(nu)})
    return {
        "constant_columns": constant,
        "likely_id_columns": likely_ids,
        "high_cardinality_columns": high_card,
    }, []


def _datetime_trends(df, cols, params, target, artifacts, cfg):
    result: dict[str, dict] = {}
    for c in _temporal_cols(df, cols):
        s = df[c].drop_nulls()
        if s.len() == 0:
            continue
        info = {"min": str(s.min()), "max": str(s.max()), "n_non_null": int(s.len())}
        try:
            bucketed = (
                df.select(pl.col(c).dt.truncate("1mo").alias("_bucket"))
                .drop_nulls()
                .group_by("_bucket")
                .len()
                .sort("_bucket")
            )
            if bucketed.height > 1:
                info["months"] = [str(v) for v in bucketed["_bucket"].to_list()]
                info["counts"] = [int(v) for v in bucketed["len"].to_list()]
        except Exception as exc:  # noqa: BLE001
            info["trend_error"] = str(exc)
        result[c] = info
    return result, []


def _target_analysis(df, cols, params, target, artifacts, cfg):
    if not target or target not in df.columns:
        return {"note": "no valid target column; target analysis skipped"}, []

    s = df[target]
    nu = s.n_unique()
    is_numeric = s.dtype in _NUMERIC
    task = "regression" if (is_numeric and nu > _CLASS_MAX_UNIQUE) else "classification"
    result: dict[str, Any] = {"target": target, "task_type": task, "n_unique": int(nu)}

    if task == "classification":
        vc = s.drop_nulls().value_counts(sort=True).head(int(cfg.get("top_k", 10)) * 2)
        count_col = "count" if "count" in vc.columns else vc.columns[-1]
        labels = vc[target].to_list()
        counts = [int(x) for x in vc[count_col].to_list()]
        total = sum(counts) or 1
        result["class_balance"] = [
            {"class": str(l), "count": n, "pct": round(n / total, 4)}
            for l, n in zip(labels, counts)
        ]
    else:
        sd = s.drop_nulls()
        result["distribution"] = {
            "mean": _clean(sd.mean()), "std": _clean(sd.std()),
            "min": _clean(sd.min()), "median": _clean(sd.median()),
            "max": _clean(sd.max()), "skew": _clean(sd.skew()),
        }
    return result, []


def _tick_summary(analysis: str, result: dict, n_plots: int) -> str:
    """One-line plain-English summary of an analysis for live CLI ticks."""
    if not isinstance(result, dict):
        return ""
    if "error" in result:
        return f"failed: {result['error']}"
    if analysis == "numeric_summary":
        n = len(result)
        return f"summarised {n} numeric column{'s' if n != 1 else ''}"
    if analysis == "distributions":
        n = len(result.get("numeric_columns", []))
        return f"{n} numeric column{'s' if n != 1 else ''} ready for distribution plots"
    if analysis == "categorical_freq":
        cols = list(result.keys())
        n = len(cols)
        if not n:
            return "no categorical columns"
        example = cols[0]
        return f"{n} categorical column{'s' if n != 1 else ''}, e.g. {example}"
    if analysis == "correlation":
        if "note" in result:
            return result["note"]
        strong = result.get("strong_pairs", [])
        if not strong:
            return "no strongly correlated pairs"
        top = max(strong, key=lambda p: abs(p["r"]))
        return f"{len(strong)} strongly correlated pair(s), top: {top['a']} ↔ {top['b']} (r={top['r']:+.2f})"
    if analysis == "outliers":
        flagged = [(c, v) for c, v in result.items() if isinstance(v, dict) and v.get("n_outliers", 0) > 0]
        if not flagged:
            return "no notable outliers"
        worst = max(flagged, key=lambda kv: kv[1]["pct_outliers"])
        return f"{len(flagged)} column(s) with outliers, worst: {worst[0]} ({worst[1]['pct_outliers']:.0%})"
    if analysis == "missingness":
        rates = (result.get("null_rates") or {})
        high = [(c, r) for c, r in rates.items() if isinstance(r, (int, float)) and r >= 0.4]
        if not high:
            return "no columns are heavily empty"
        worst = max(high, key=lambda kv: kv[1])
        return f"{len(high)} column(s) ≥40% null, worst: {worst[0]} ({worst[1]:.0%})"
    if analysis == "cardinality":
        c = result.get("constant_columns", [])
        i = result.get("likely_id_columns", [])
        h = result.get("high_cardinality_columns", [])
        parts = []
        if c: parts.append(f"{len(c)} constant")
        if i: parts.append(f"{len(i)} likely IDs")
        if h: parts.append(f"{len(h)} high-cardinality")
        return ", ".join(parts) if parts else "nothing unusual"
    if analysis == "datetime_trends":
        n = len(result)
        return f"{n} date column{'s' if n != 1 else ''} with usable time range"
    if analysis == "target_analysis":
        if "note" in result:
            return result["note"]
        task = result.get("task_type", "?")
        if task == "classification":
            balance = result.get("class_balance", [])
            top = " vs ".join(f"{b['pct']:.0%}" for b in balance[:2])
            return f"classification target, {top} split"
        dist = result.get("distribution", {})
        return f"regression target, mean≈{dist.get('mean'):.3g}, skew {dist.get('skew'):+.2f}" if dist.get("mean") is not None else "regression target"
    return ""


_DISPATCH = {
    "numeric_summary": _numeric_summary,
    "distributions": _distributions,
    "categorical_freq": _categorical_freq,
    "correlation": _correlation,
    "outliers": _outliers,
    "missingness": _missingness,
    "cardinality": _cardinality,
    "datetime_trends": _datetime_trends,
    "target_analysis": _target_analysis,
}


def run_analyses(
    plan: list[dict],
    frames: dict[str, pl.DataFrame],
    target_table: str | None,
    target_column: str | None,
    artifacts: ArtifactStore,
    cfg: dict | None = None,
    progress=None,
) -> tuple[dict, list[dict]]:
    """Execute the analysis plan. Returns (results, plots).

    results: {table: {analysis: result_or_error}}
    plots:   [{title, path, kind, table, analysis}]
    Failures are captured per (analysis, table) as {"error": ...}, never raised.

    `progress` (optional) is called after each (table, analysis) completes with
    {"table","analysis","ok","n_plots","summary","error"} — used by the CLI to
    stream live ticks. Callback exceptions are swallowed so progress never aborts
    the loop.
    """
    cfg = cfg or {}
    results: dict[str, dict] = {}
    all_plots: list[dict] = []

    for item in plan:
        analysis = item.get("analysis")
        table = item.get("table")
        cols = item.get("columns") or None
        params = item.get("params") or {}
        fn = _DISPATCH.get(analysis)
        if fn is None or table not in frames:
            continue
        tgt = target_column if table == target_table else None
        local_cfg = {**cfg, "_prefix": f"{table.split('.')[-1]}_"}
        try:
            res, made = fn(frames[table], cols, params, tgt, artifacts, local_cfg)
        except Exception as exc:  # noqa: BLE001
            res, made = {"error": str(exc)}, []
        results.setdefault(table, {})[analysis] = res
        for p in made:
            all_plots.append({**p, "table": table, "analysis": analysis})
        if progress is not None:
            try:
                ok = "error" not in res
                progress({
                    "table": table, "analysis": analysis, "ok": ok,
                    "n_plots": len(made),
                    "summary": _tick_summary(analysis, res, len(made)),
                    "error": (None if ok else res.get("error")),
                })
            except Exception:  # noqa: BLE001
                pass

    return results, all_plots
