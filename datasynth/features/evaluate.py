"""Score a feature matrix to drive the refinement loop.

Fits a quick RandomForest baseline for a cross-validated metric + feature
importances, augmented with mutual information, per-feature null/variance/target
correlation, redundancy clusters, and leakage flags. The whole sklearn path is
guarded; on any failure it degrades to a stats-only scorecard so the agentic loop
never crashes on a pathological matrix.

Returns a JSON-serializable scorecard:
  {task, metric_name, metric_value, n_features, n_rows,
   features: [{name, importance, mutual_info, null_rate, corr_target, leakage}],
   redundancy_clusters: [[...]], leakage_features: [...], error?}
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from datasynth._dtypes import CLASS_MAX_UNIQUE as _CLASS_MAX_UNIQUE
from datasynth._dtypes import NUMERIC as _NUMERIC
from datasynth._dtypes import TEMPORAL as _TEMPORAL

_LEAK_IMPORTANCE = 0.9
_LEAK_CORR = 0.98
_REDUNDANT_CORR = 0.95


def _feature_columns(df: pl.DataFrame, key: str, target: str) -> list[str]:
    drop = {key, target}
    return [
        c for c in df.columns
        if c not in drop and df[c].dtype not in _TEMPORAL
    ]


def _encode_X(df: pl.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Numeric matrix for sklearn: numerics imputed by median, categoricals factorized."""
    import pandas as pd

    pdf = df.select(cols).to_pandas()
    out = {}
    for c in cols:
        s = pdf[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            col = s.astype("float64")
            med = col.median()
            out[c] = col.fillna(med if not math.isnan(med) else 0.0)
        else:
            codes = pd.factorize(s, use_na_sentinel=True)[0].astype("float64")
            out[c] = pd.Series(codes, index=s.index)  # -1 marks NA
    X = pd.DataFrame(out, columns=cols)
    return X.to_numpy(dtype="float64"), cols


def _encode_y(df: pl.DataFrame, target: str):
    import pandas as pd

    y = df[target].to_pandas()
    mask = y.notna().to_numpy()
    return y, mask


def _task_type(df: pl.DataFrame, target: str) -> str:
    is_num = df[target].dtype in _NUMERIC
    return "regression" if (is_num and df[target].n_unique() > _CLASS_MAX_UNIQUE) else "classification"


def _stats_only(df, cols, target, task, n_rows, note) -> dict:
    feats = []
    tnum = df[target].cast(pl.Float64, strict=False) if task == "classification" else df[target]
    for c in cols:
        null_rate = round(df[c].null_count() / n_rows, 4) if n_rows else 0.0
        corr = None
        if df[c].dtype in _NUMERIC:
            try:
                corr = df.select(pl.corr(pl.col(c).cast(pl.Float64), tnum.cast(pl.Float64))).item()
            except Exception:
                corr = None
        feats.append({"name": c, "importance": None, "mutual_info": None,
                      "null_rate": null_rate,
                      "corr_target": round(corr, 4) if corr is not None and not math.isnan(corr) else None,
                      "leakage": bool(corr is not None and abs(corr) >= _LEAK_CORR)})
    return {
        "task": task, "metric_name": None, "metric_value": None,
        "n_features": len(cols), "n_rows": n_rows, "features": feats,
        "redundancy_clusters": [], "leakage_features": [f["name"] for f in feats if f["leakage"]],
        "error": note,
    }


def score_features(df: pl.DataFrame, target: str, key: str, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    n_rows = df.height
    cols = _feature_columns(df, key, target)
    task = _task_type(df, target)

    if not cols or n_rows < 10:
        return _stats_only(df, cols, target, task, n_rows, "too few rows/features for a model")

    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder

        X, names = _encode_X(df, cols)
        y_series, mask = _encode_y(df, target)
        X = X[mask]
        y_raw = y_series[mask].to_numpy()

        if task == "classification":
            y = LabelEncoder().fit_transform(y_raw)
            n_classes = len(np.unique(y))
            model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
            if n_classes == 2:
                metric_name, scoring = "roc_auc", "roc_auc"
            else:
                metric_name, scoring = "accuracy", "accuracy"
            mi = mutual_info_classif(X, y, random_state=0)
        else:
            y = y_raw.astype("float64")
            model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
            metric_name, scoring = "r2", "r2"
            mi = mutual_info_regression(X, y, random_state=0)

        folds = 3 if X.shape[0] >= 30 else 2
        try:
            cv = cross_val_score(model, X, y, cv=folds, scoring=scoring, n_jobs=-1)
            metric_value = float(np.nanmean(cv))
        except Exception:
            metric_value = None

        model.fit(X, y)
        importances = model.feature_importances_

        # per-feature null + target correlation
        tnum = df[target].cast(pl.Float64, strict=False)
        feats = []
        for i, c in enumerate(names):
            null_rate = round(df[c].null_count() / n_rows, 4) if n_rows else 0.0
            corr = None
            if df[c].dtype in _NUMERIC:
                try:
                    corr = df.select(pl.corr(pl.col(c).cast(pl.Float64), tnum)).item()
                except Exception:
                    corr = None
            imp = float(importances[i])
            leak = imp >= _LEAK_IMPORTANCE or (corr is not None and not math.isnan(corr) and abs(corr) >= _LEAK_CORR)
            feats.append({
                "name": c, "importance": round(imp, 4),
                "mutual_info": round(float(mi[i]), 4),
                "null_rate": null_rate,
                "corr_target": round(corr, 4) if corr is not None and not math.isnan(corr) else None,
                "leakage": bool(leak),
            })
        feats.sort(key=lambda f: f["importance"] or 0.0, reverse=True)

        # redundancy: high pairwise correlation among encoded features
        clusters = _redundancy_clusters(X, names)

        return {
            "task": task, "metric_name": metric_name,
            "metric_value": round(metric_value, 4) if metric_value is not None else None,
            "n_features": len(names), "n_rows": int(X.shape[0]),
            "features": feats, "redundancy_clusters": clusters,
            "leakage_features": [f["name"] for f in feats if f["leakage"]],
        }
    except Exception as exc:  # noqa: BLE001
        return _stats_only(df, cols, target, task, n_rows, f"sklearn scoring failed: {exc}")


def _redundancy_clusters(X: np.ndarray, names: list[str]) -> list[list[str]]:
    try:
        if X.shape[1] < 2:
            return []
        with np.errstate(all="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        seen, clusters = set(), []
        for i in range(len(names)):
            if i in seen:
                continue
            group = [names[i]]
            for j in range(i + 1, len(names)):
                if j in seen:
                    continue
                r = corr[i, j]
                if not math.isnan(r) and abs(r) >= _REDUNDANT_CORR:
                    group.append(names[j])
                    seen.add(j)
            if len(group) > 1:
                clusters.append(group)
                seen.add(i)
        return clusters
    except Exception:
        return []
