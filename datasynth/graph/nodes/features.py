"""Feature-generation phase nodes (agentic refinement loop).

Flow: feat_plan → feat_build → feat_evaluate → feat_critique
        ↑__________________________________________|  (loop while refining)
      feat_critique → feat_review (interrupt) → feat_revise → feat_build

plan/critique/revise decide *what* (LLM); feat_build assembles the entity-grain
matrix via datasynth.features.sql_build + transforms, feat_evaluate scores it with a
sklearn baseline. The loop refines features until the agent says "done", the metric
stops improving, or feat_max_iterations is hit. feat_review/revise reuse the generic
gate in datasynth.graph.review.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from datasynth.clean.store import FrameStore
from datasynth.datasource.base import DataSource
from datasynth.features import sql_build as sb
from datasynth.features.evaluate import score_features
from datasynth.features.transforms import apply_encoders, apply_transforms
from datasynth.features.vocabulary import (
    AGG_FUNCS,
    ENCODE_VOCABULARY,
    TRANSFORM_VOCABULARY,
)
from datasynth.graph.review import ReviewDecision, make_review_node, make_revise_node
from datasynth.graph.state import AgentState

_MATRIX_KEY = "__features__"

_AGG_DOC = "\n".join(f"    - {n}: {d}" for n, d in AGG_FUNCS.items())
_TF_DOC = "\n".join(f"    - {n}: {d}" for n, d in TRANSFORM_VOCABULARY.items())
_ENC_DOC = "\n".join(f"    - {n}: {d}" for n, d in ENCODE_VOCABULARY.items())

_SPEC_DOC = f"""\
A feature spec has a `kind` and the fields that kind needs:
  - base: passthrough an entity-table column. fields: column (= name)
  - aggregate: aggregate a child table onto the entity. fields: child_table
    (schema.table), agg_fn, source_column (omit for count), name. Allowed agg_fn:
{_AGG_DOC}
  - transform: derive a feature in-memory from already-built features. fields: op,
    name, params_json (JSON object). Ops:
{_TF_DOC}
  - encode: encode a categorical column. fields: method, column, name. Methods:
{_ENC_DOC}
The join key (FK→entity) is resolved automatically; do not specify it. Avoid target
leakage: never build features from the target or from post-outcome columns."""

_PLAN_SYSTEM = f"""\
You are a senior data scientist engineering features for a supervised model. Using the
problem statement, the target, profiles, EDA findings, the cleaning log, and the
foreign-key graph, design a STRONG initial feature set for a model at the entity grain.

Think like a practitioner: aggregate behavioral child tables (counts, sums, means,
recency, stddev) onto the entity; add ratios/interactions/tenure where they encode
signal; encode useful categoricals. Prefer features the EDA findings suggest matter.

{_SPEC_DOC}
"""

_CRITIQUE_SYSTEM = f"""\
You are refining a feature set based on a scorecard from a baseline model (feature
importances, mutual information, leakage flags, redundancy clusters, and a cross-
validated metric). Act like a data scientist iterating toward a better model.

Decide an action:
  - "refine": return the COMPLETE revised spec list — drop weak/redundant/leaky
    features, and ADD new ones worth trying (interactions of strong features, more
    aggregations, alternative encodings). Explain each material change.
  - "done": the feature set is good and further changes show diminishing returns.

Drop any feature flagged as leakage. Keep the set focused, not bloated.

{_SPEC_DOC}
"""

_REVISE_SYSTEM = f"""\
You are adjusting a feature set based on the user's comment. The user may ask to add,
drop, or change features, or to keep iterating.

Choose an action:
  - "proceed": the user approved — keep the current feature set.
  - "revise": return the COMPLETE revised spec list reflecting the comment.

{_SPEC_DOC}
"""


class FeatureSpec(BaseModel):
    kind: Literal["base", "aggregate", "transform", "encode"]
    name: str = Field(description="Output feature name (for base = the column name)")
    column: str | None = Field(default=None, description="Entity/categorical column (base, encode)")
    child_table: str | None = Field(default=None, description="Child table (aggregate)")
    agg_fn: str | None = Field(default=None, description=f"Aggregate fn: {', '.join(AGG_FUNCS)}")
    source_column: str | None = Field(default=None, description="Column to aggregate (aggregate)")
    op: str | None = Field(default=None, description=f"Transform op: {', '.join(TRANSFORM_VOCABULARY)}")
    method: str | None = Field(default=None, description=f"Encoder: {', '.join(ENCODE_VOCABULARY)}")
    params_json: str = Field(default="{}", description='Transform params as JSON, e.g. {"input_a":"x","input_b":"y"}')
    reason: str = Field(default="", description="Why this feature is useful")


class FeaturePlan(BaseModel):
    reasoning: str = Field(description="Overall feature-engineering rationale")
    specs: list[FeatureSpec] = Field(default_factory=list)


class FeatureCritique(BaseModel):
    action: Literal["refine", "done"] = Field(default="done")
    reasoning: str = Field(description="What you changed and why, or why the set is final")
    specs: list[FeatureSpec] = Field(default_factory=list, description="Complete revised list (when action='refine')")


class FeatureRevision(ReviewDecision):
    action: Literal["proceed", "revise"] = Field(default="proceed")
    specs: list[FeatureSpec] = Field(default_factory=list)


def _normalize(specs: list[FeatureSpec]) -> list[dict]:
    out: list[dict] = []
    for s in specs:
        try:
            params = json.loads(s.params_json or "{}")
            if not isinstance(params, dict):
                params = {}
        except (ValueError, TypeError):
            params = {}
        out.append({
            "kind": s.kind, "name": s.name, "column": s.column,
            "child_table": s.child_table, "agg_fn": s.agg_fn,
            "source_column": s.source_column, "op": s.op, "method": s.method,
            "params": params, "reason": s.reason,
        })
    return out


def _spec_names(specs: list[dict]) -> list[str]:
    return [s.get("name") or s.get("column") or s.get("op") or s.get("method") or "?" for s in specs]


def _profiles_brief(profiles: dict) -> str:
    lines = []
    for table, p in profiles.items():
        if "error" in p:
            continue
        cols = p.get("columns", {})
        bits = [f"{c} ({s.get('dtype')})" for c, s in cols.items()]
        lines.append(f"{table}: " + ", ".join(bits))
    return "\n".join(lines)


def _fk_brief(schema, entity_table, links) -> str:
    lines = [f"Entity: {entity_table} (key {sb.entity_key(schema, entity_table)})"]
    for child, fk in links.items():
        t = schema.table(child)
        cols = ", ".join(c.name for c in t.columns) if t else ""
        lines.append(f"Child {child} (fk {fk}): {cols}")
    return "\n".join(lines)


# ── propose ──────────────────────────────────────────────────────────────────

def make_feat_plan_node(model: BaseChatModel, cfg: dict):
    selector = model.with_structured_output(FeaturePlan)

    async def feat_plan(state: AgentState, config: RunnableConfig) -> dict:
        ds: DataSource = config["configurable"]["datasource"]
        schema = ds.introspect_schema()
        entity_table = state.get("target_table") or ""
        key = sb.entity_key(schema, entity_table) if entity_table else None
        links = sb.child_links(schema, entity_table, state.get("selected_tables", [])) if entity_table else {}

        context = (
            f"Problem: {state.get('ask','')}\n"
            f"Target: {state.get('target_column') or '(none)'}\n\n"
            f"Foreign-key graph:\n{_fk_brief(schema, entity_table, links)}\n\n"
            f"EDA summary: {state.get('eda_summary','')}\n"
            f"EDA findings: {json.dumps(state.get('eda_findings', []), default=str)}\n"
            f"Cleaning log: {json.dumps(state.get('cleaning_log', []), default=str)}\n\n"
            f"Profiles:\n{_profiles_brief(state.get('profiles', {}))}"
        )
        messages = [SystemMessage(content=_PLAN_SYSTEM), HumanMessage(content=context)]
        result: FeaturePlan | None = None
        async for partial in selector.astream(messages, config=config):
            result = partial  # type: ignore[assignment]
        if result is None:
            raise RuntimeError("LLM returned no output for feature plan.")

        return {
            "entity_table": entity_table,
            "entity_key": key or "",
            "feature_plan": _normalize(result.specs),
            "feat_reasoning": result.reasoning,
            "feat_iteration": 0,
            "feat_history": [],
            "feat_max_iterations": cfg.get("max_iterations", 4),
            "feat_metric_epsilon": cfg.get("epsilon", 0.005),
            "phase": "feature_gen",
        }

    return feat_plan


# ── deterministic build ──────────────────────────────────────────────────────

def make_feat_build_node(cfg: dict):
    def feat_build(state: AgentState, config: RunnableConfig) -> dict:
        ds: DataSource = config["configurable"]["datasource"]
        store: FrameStore = config["configurable"]["frame_store"]
        schema = ds.introspect_schema()

        entity_table = state.get("entity_table", "")
        key = state.get("entity_key", "")
        target = state.get("target_column") or ""
        specs = state.get("feature_plan", [])

        if not entity_table or not key:
            return {"feature_matrix_meta": {"error": "no entity table/key resolved"},
                    "feat_build_log": []}

        base_cols = [s["column"] for s in specs if s["kind"] == "base" and s.get("column")]
        if target:
            base_cols.append(target)
        agg_specs = [s for s in specs if s["kind"] == "aggregate"]
        tf_specs = [{"op": s["op"], "name": s["name"], "params": s["params"]}
                    for s in specs if s["kind"] == "transform"]
        enc_specs = [{"method": s["method"], "column": s["column"], "name": s["name"]}
                     for s in specs if s["kind"] == "encode"]

        links = sb.child_links(schema, entity_table, state.get("selected_tables", []))
        try:
            sql, accepted, skipped = sb.build_matrix_sql(
                schema, entity_table, key, base_cols, agg_specs, links,
                int(cfg.get("sample_size", 10000)),
            )
            df = sb.assemble(ds, sql, max_rows=int(cfg.get("max_rows", 10_000_000)))
        except Exception as exc:  # noqa: BLE001
            return {"feature_matrix_meta": {"error": f"matrix build failed: {exc}"},
                    "feat_build_log": []}

        df, tlog = apply_transforms(df, tf_specs)
        df, elog = apply_encoders(df, enc_specs, target)
        store.put(_MATRIX_KEY, df)

        feature_cols = [c for c in df.columns if c not in (key, target)]
        meta = {
            "n_rows": df.height, "n_features": len(feature_cols),
            "columns": feature_cols,
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "accepted_aggregates": accepted,
        }
        return {"feature_matrix_meta": meta, "feat_build_log": skipped + tlog + elog}

    return feat_build


# ── deterministic evaluate ──────────────────────────────────────────────────

def make_feat_evaluate_node(cfg: dict):
    def feat_evaluate(state: AgentState, config: RunnableConfig) -> dict:
        store: FrameStore = config["configurable"]["frame_store"]
        key = state.get("entity_key", "")
        target = state.get("target_column") or ""
        meta = state.get("feature_matrix_meta", {})

        if "error" in meta or not store.has(_MATRIX_KEY) or not target:
            return {"feature_scores": {"error": meta.get("error", "no feature matrix/target")},
                    "baseline_metric": {}, "feat_eval_summary": "evaluation skipped"}

        scorecard = score_features(store.get(_MATRIX_KEY), target, key, cfg)
        metric = {"name": scorecard.get("metric_name"), "value": scorecard.get("metric_value")}
        summary = (
            f"{scorecard.get('task')} · {metric['name']}="
            f"{metric['value']} · {scorecard.get('n_features')} features · "
            f"{len(scorecard.get('leakage_features', []))} leakage flag(s)"
        )
        return {"feature_scores": scorecard, "baseline_metric": metric,
                "feat_eval_summary": summary}

    return feat_evaluate


# ── critique (agentic loop driver) ──────────────────────────────────────────

def _scorecard_brief(sc: dict) -> str:
    if "error" in sc:
        return f"scoring error: {sc['error']}"
    top = sc.get("features", [])[:15]
    rows = [f"  {f['name']}: imp={f['importance']} mi={f['mutual_info']} "
            f"null={f['null_rate']} corr={f['corr_target']}"
            + (" LEAK" if f["leakage"] else "") for f in top]
    return (
        f"task={sc.get('task')} {sc.get('metric_name')}={sc.get('metric_value')} "
        f"n_features={sc.get('n_features')}\n"
        f"leakage={sc.get('leakage_features')}\n"
        f"redundant_clusters={sc.get('redundancy_clusters')}\n"
        "feature scores (top by importance):\n" + "\n".join(rows)
    )


def make_feat_critique_node(model: BaseChatModel):
    selector = model.with_structured_output(FeatureCritique)

    async def feat_critique(state: AgentState, config: RunnableConfig) -> dict:
        scorecard = state.get("feature_scores", {})
        context = (
            f"Problem: {state.get('ask','')}\n"
            f"Target: {state.get('target_column')}\n"
            f"Iteration: {state.get('feat_iteration', 0)} / {state.get('feat_max_iterations', 4)}\n"
            f"Metric history: {json.dumps(state.get('feat_history', []), default=str)}\n\n"
            f"Current specs:\n{json.dumps(state.get('feature_plan', []), default=str)}\n\n"
            f"Scorecard:\n{_scorecard_brief(scorecard)}"
        )
        messages = [SystemMessage(content=_CRITIQUE_SYSTEM), HumanMessage(content=context)]
        result: FeatureCritique | None = None
        async for partial in selector.astream(messages, config=config):
            result = partial  # type: ignore[assignment]
        if result is None:
            raise RuntimeError("LLM returned no output for feature critique.")

        prev = state.get("feature_plan", [])
        new_plan = _normalize(result.specs) if (result.action == "refine" and result.specs) else prev
        prev_names, new_names = set(_spec_names(prev)), set(_spec_names(new_plan))
        changes = {
            "added": sorted(new_names - prev_names),
            "dropped": sorted(prev_names - new_names),
            "kept": len(prev_names & new_names),
        }
        iteration = state.get("feat_iteration", 0) + 1
        history = state.get("feat_history", []) + [{
            "iteration": iteration,
            "metric": state.get("baseline_metric", {}).get("value"),
            "n_features": state.get("feature_matrix_meta", {}).get("n_features"),
            "action": result.action,
        }]
        return {
            "feature_plan": new_plan,
            "feat_reasoning": result.reasoning,
            "feat_action": result.action,
            "feat_changes": changes,
            "feat_iteration": iteration,
            "feat_history": history,
        }

    return feat_critique


def route_after_critique(state: AgentState) -> str:
    """Loop back to build while the agent wants to refine, under the iteration cap,
    and the metric is still improving by at least epsilon."""
    if state.get("feat_action") != "refine":
        return "review"
    if state.get("feat_iteration", 0) >= state.get("feat_max_iterations", 4):
        return "review"
    hist = state.get("feat_history", [])
    eps = state.get("feat_metric_epsilon", 0.005)
    metrics = [h["metric"] for h in hist if h.get("metric") is not None]
    if len(metrics) >= 2 and (metrics[-1] - metrics[-2]) < eps:
        return "review"  # diminishing returns
    return "refine"


# ── gate nodes (reuse the generic review gate) ──────────────────────────────

feat_review_node = make_review_node(
    lambda state: {
        "kind": "features",
        "feature_matrix_meta": state.get("feature_matrix_meta", {}),
        "feature_scores": state.get("feature_scores", {}),
        "baseline_metric": state.get("baseline_metric", {}),
        "feat_eval_summary": state.get("feat_eval_summary", ""),
        "feat_history": state.get("feat_history", []),
    }
)


def _revise_context(state: AgentState) -> str:
    return (
        f"Current specs:\n{json.dumps(state.get('feature_plan', []), default=str)}\n\n"
        f"Scorecard:\n{_scorecard_brief(state.get('feature_scores', {}))}"
    )


def _revise_apply(state: AgentState, result: FeatureRevision) -> dict:
    plan = _normalize(result.specs) if result.specs else state.get("feature_plan", [])
    return {"feature_plan": plan}


def make_feat_revise_node(model: BaseChatModel):
    return make_revise_node(
        model,
        system_prompt=_REVISE_SYSTEM,
        schema_model=FeatureRevision,
        build_context=_revise_context,
        apply=_revise_apply,
    )
