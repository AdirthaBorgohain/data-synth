"""EDA phase nodes.

Flow: eda_propose (agentic) → eda_run (deterministic) → eda_synthesize (agentic)
      → eda_review (interrupt) → eda_revise (agentic, optional) → eda_run ...

propose/synthesize/revise decide *what* (LLM); eda_run executes the analysis plan
deterministically via datasynth.eda.run_analyses on frames held in the FrameStore.
eda_review / eda_revise reuse the generic gate in datasynth.graph.review.

Target column is chosen upstream by the target-definition phase; eda nodes read
it from state and never modify it.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.callbacks import adispatch_custom_event, dispatch_custom_event
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from datasynth.artifacts.store import ArtifactStore
from datasynth.clean.store import FrameStore
from datasynth.eda.analyses import EDA_VOCABULARY, run_analyses
from datasynth.graph.nodes._helpers import profiles_brief as _profiles_brief
from datasynth.graph.review import ReviewDecision, make_review_node, make_revise_node
from datasynth.graph.state import AgentState

_EDA_DOC = "\n".join(f"  - {name}: {desc}" for name, desc in EDA_VOCABULARY.items())

_PROPOSE_SYSTEM = f"""\
You are an exploratory-data-analysis agent. The data has been profiled and cleaned,
and the target column has already been chosen and materialized in the entity table.
Plan an EDA pass that will inform downstream feature engineering and modeling.

Pick analyses ONLY from this menu:
{_EDA_DOC}

Each analysis targets one table (qualified schema.table). Leave `columns` empty to
let the analysis pick relevant columns automatically, or name specific columns.
Prefer a broad pass (numeric_summary, distributions, correlation, missingness,
outliers, cardinality) plus target_analysis when a target is set. Give a short
`reason` per analysis.
"""

_REVISE_SYSTEM = f"""\
You are adjusting an EDA plan based on the user's comment. The user may want to
add or drop analyses. The target is fixed at this point — to change the target,
the user must restart from the target-definition phase.

Analyses available:
{_EDA_DOC}

Choose an action:
  - "proceed": the user approved as-is — keep the current plan.
  - "revise": return the COMPLETE revised analysis list.
"""


class EdaAnalysis(BaseModel):
    analysis: str = Field(description=f"One of: {', '.join(EDA_VOCABULARY)}")
    table: str = Field(description="Qualified table name (schema.table)")
    columns: list[str] = Field(
        default_factory=list, description="Specific columns, or empty for auto-select"
    )
    params_json: str = Field(
        default="{}",
        description='Analysis settings as a JSON object string, e.g. {"top_k":10}',
    )
    reason: str = Field(description="Why this analysis is useful")


class EdaPlan(BaseModel):
    reasoning: str = Field(description="Overall rationale for the EDA plan")
    analyses: list[EdaAnalysis] = Field(default_factory=list)


class EdaRevision(ReviewDecision):
    action: Literal["proceed", "revise"] = Field(default="proceed")
    analyses: list[EdaAnalysis] = Field(
        default_factory=list, description="Complete revised analysis list (when action='revise')"
    )


class Finding(BaseModel):
    topic: str = Field(description="Short topic, e.g. 'class imbalance' or 'age outliers'")
    observation: str = Field(description="What the analysis showed, with concrete numbers")
    downstream_implication: str = Field(
        description="What this means for cleaning / feature engineering / modeling"
    )
    table: str | None = Field(
        default=None,
        description=("Qualified table (schema.table) of the EDA result that best "
                     "evidences this finding. Must match a key in the EDA results "
                     "JSON. Leave null if no single result fits."),
    )
    analysis: str | None = Field(
        default=None,
        description=("Analysis name from EDA_VOCABULARY that best evidences this "
                     "finding (e.g. correlation, outliers, target_analysis). "
                     "Leave null if no single result fits."),
    )


class EdaFindings(BaseModel):
    summary: str = Field(description="2-4 sentence overview of the data's character")
    findings: list[Finding] = Field(default_factory=list)


def _validate_analyses(state: AgentState, analyses: list[EdaAnalysis]) -> list[dict]:
    """Drop analyses referencing unknown tables/names/columns; parse params."""
    selected = set(state.get("selected_tables", []))
    profiles = state.get("profiles", {})
    valid: list[dict] = []
    for a in analyses:
        if a.table not in selected or a.analysis not in EDA_VOCABULARY:
            continue
        cols = set(profiles.get(a.table, {}).get("columns", {}))
        columns = [c for c in (a.columns or []) if c in cols]
        try:
            params = json.loads(a.params_json or "{}")
        except (ValueError, TypeError):
            params = {}
        if not isinstance(params, dict):
            params = {}
        valid.append(
            {"analysis": a.analysis, "table": a.table, "columns": columns,
             "params": params, "reason": a.reason}
        )
    return valid


def _validate_target(state: AgentState, table: str | None, column: str | None):
    """Keep the target only if it points at a real selected table + column."""
    selected = set(state.get("selected_tables", []))
    profiles = state.get("profiles", {})
    if table not in selected or column is None:
        return None, None
    if column not in profiles.get(table, {}).get("columns", {}):
        return None, None
    return table, column


# ── propose ──────────────────────────────────────────────────────────────────

def make_eda_propose_node(model: BaseChatModel):
    selector = model.with_structured_output(EdaPlan)

    async def eda_propose(state: AgentState, config: RunnableConfig) -> dict:
        context = (
            f"User ask: {state.get('ask', '')}\n\n"
            f"Cleaned profiles:\n{_profiles_brief(state.get('profiles', {}))}"
        )
        messages = [
            SystemMessage(content=_PROPOSE_SYSTEM),
            HumanMessage(content=context),
        ]
        result: EdaPlan | None = None
        async for partial in selector.astream(messages, config=config):
            result = partial  # type: ignore[assignment]
        if result is None:
            raise RuntimeError("LLM returned no output for EDA plan.")

        # Re-validate the target from state in case the entity table changed.
        tt, tc = _validate_target(state, state.get("target_table"), state.get("target_column"))
        return {
            "eda_plan": _validate_analyses(state, result.analyses),
            "eda_reasoning": result.reasoning,
            "target_table": tt or "",
            "target_column": tc or "",
        }

    return eda_propose


# ── deterministic run ────────────────────────────────────────────────────────

def make_eda_run_node(cfg: dict):
    def eda_run(state: AgentState, config: RunnableConfig) -> dict:
        store: FrameStore = config["configurable"]["frame_store"]
        artifacts: ArtifactStore = config["configurable"]["artifacts"]

        frames = {
            t: store.get(t)
            for t in state.get("selected_tables", [])
            if store.has(t)
        }

        # Live tick per (table, analysis) finishing — picked up by the runner as
        # on_custom_event and rendered as a one-liner so the CLI is no longer silent
        # while eda_run executes.
        def _progress(payload: dict) -> None:
            try:
                dispatch_custom_event("eda_analysis", payload)
            except Exception:  # noqa: BLE001
                pass

        results, plots = run_analyses(
            state.get("eda_plan", []),
            frames,
            state.get("target_table"),
            state.get("target_column"),
            artifacts,
            cfg,
            progress=_progress,
        )
        return {"eda_results": results, "eda_plots": plots, "phase": "eda"}

    return eda_run


# ── synthesize findings ──────────────────────────────────────────────────────

def make_eda_synthesize_node(model: BaseChatModel):
    selector = model.with_structured_output(EdaFindings)

    async def eda_synthesize(state: AgentState, config: RunnableConfig) -> dict:
        context = (
            f"User ask: {state.get('ask', '')}\n"
            f"Target: {state.get('target_column') or '(none)'} "
            f"in {state.get('target_table') or '(none)'}\n\n"
            f"EDA results (JSON):\n{json.dumps(state.get('eda_results', {}), default=str)}"
        )
        messages = [
            SystemMessage(content=(
                "You are summarizing EDA results into findings that guide downstream "
                "feature engineering and modeling. Your reader is a smart non-"
                "technical stakeholder — every finding must read like a short paragraph "
                "of plain English, not shorthand or notes.\n\n"
                "Hard rules for each finding:\n"
                "  • `topic`: a noun phrase of 3–8 words naming what was found "
                "(e.g. 'Class imbalance in the churn label').\n"
                "  • `observation`: 2–3 complete sentences. State WHAT was found, with "
                "concrete numbers and the columns/tables involved, and add enough "
                "context so the sentence stands alone. Never use arrows (→, ->), "
                "fragments, or telegraphic shorthand. Spell out comparisons in full "
                "(say 'about 60% are active while 40% are inactive', not "
                "'60% active -> inactive').\n"
                "  • `downstream_implication`: 1–2 complete sentences explaining WHY it "
                "matters for cleaning, feature engineering, or modeling, and a "
                "concrete recommended action.\n"
                "  • `table` and `analysis` MUST be set to the EDA result that "
                "evidences the finding. `table` is a key of the EDA results JSON; "
                "`analysis` is one of: numeric_summary, distributions, "
                "categorical_freq, correlation, outliers, missingness, cardinality, "
                "datetime_trends, target_analysis. Leave both null ONLY if truly no "
                "single result fits.\n\n"
                "Never invent numbers or columns that aren't in the EDA results. "
                "Aim for 4–8 findings, ordered by impact on the model.\n\n"
                "Examples of the right style:\n\n"
                "  GOOD finding (target_analysis on public.customers):\n"
                "    topic: 'Imbalanced churn label'\n"
                "    observation: 'Out of 1,000 sampled customers in public.customers, "
                "about 87% are still active and only 13% have churned. This is a "
                "moderately imbalanced binary target with churn as the minority class.'\n"
                "    downstream_implication: 'Accuracy will be misleading on this "
                "split, so evaluate with ROC-AUC or PR-AUC and consider class "
                "weighting or stratified sampling during training.'\n\n"
                "  GOOD finding (correlation on public.usage_monthly):\n"
                "    topic: 'Active users and API requests move together'\n"
                "    observation: 'In public.usage_monthly the columns active_users "
                "and api_requests have a Pearson correlation of 0.92, indicating they "
                "carry almost the same signal across months.'\n"
                "    downstream_implication: 'Keeping both will inflate variance and "
                "may destabilise linear models. Drop one or replace them with a "
                "derived ratio such as api_requests_per_active_user.'\n\n"
                "  BAD finding (do NOT do this):\n"
                "    topic: 'churn'\n"
                "    observation: '60% active -> inactive.'\n"
                "    downstream_implication: '→ AUC + weights.'\n"
                "  (Reasons it's bad: fragments, arrows, ambiguous numbers, no table "
                "named, action not spelled out.)"
            )),
            HumanMessage(content=context),
        ]
        # Stream findings live: emit the summary once it first becomes available,
        # then dispatch each finding the moment all its fields look complete. The
        # final state still carries the consolidated lists for downstream phases.
        result: EdaFindings | None = None
        summary_emitted = False
        seen = 0
        async for partial in selector.astream(messages, config=config):
            result = partial  # type: ignore[assignment]
            if not summary_emitted and getattr(partial, "summary", ""):
                try:
                    await adispatch_custom_event("eda_summary", {"text": partial.summary})
                except Exception:  # noqa: BLE001
                    pass
                summary_emitted = True
            findings = list(getattr(partial, "findings", []) or [])
            while seen < len(findings):
                f = findings[seen]
                topic = (getattr(f, "topic", "") or "").strip()
                obs = (getattr(f, "observation", "") or "").strip()
                impl = (getattr(f, "downstream_implication", "") or "").strip()
                # only emit when the finding looks complete enough to read,
                # AND only if there's at least one more finding behind it
                # (or this is the final partial) so we don't print mid-token.
                is_last = seen == len(findings) - 1
                ready = topic and obs and (impl or not is_last)
                if not ready:
                    break
                try:
                    await adispatch_custom_event("eda_finding", f.model_dump())
                except Exception:  # noqa: BLE001
                    pass
                seen += 1
        if result is None:
            raise RuntimeError("LLM returned no output for EDA findings.")
        # Emit any trailing finding that wasn't flushed (final element of the stream).
        findings = list(result.findings or [])
        while seen < len(findings):
            try:
                await adispatch_custom_event("eda_finding", findings[seen].model_dump())
            except Exception:  # noqa: BLE001
                pass
            seen += 1
        return {
            "eda_summary": result.summary,
            "eda_findings": [f.model_dump() for f in result.findings],
        }

    return eda_synthesize


# ── render finding-cited plots ───────────────────────────────────────────────

def make_eda_render_node(cfg: dict):
    """Sync node that materialises one annotated PNG per finding that cites a
    plottable EDA result. Streams `eda_plot_saved` events as each PNG is written,
    so the CLI can link each path beside the finding the user just read."""

    from datasynth.eda.render import render_finding

    max_plots = int(cfg.get("max_plots", 12))

    def eda_render(state: AgentState, config: RunnableConfig) -> dict:
        artifacts: ArtifactStore = config["configurable"]["artifacts"]
        store: FrameStore = config["configurable"]["frame_store"]
        findings = list(state.get("eda_findings", []) or [])
        eda_results = state.get("eda_results", {}) or {}

        rendered_findings: list[dict] = []
        plots_meta: list[dict] = []
        rendered_count = 0
        for idx, finding in enumerate(findings, start=1):
            path = None
            if rendered_count < max_plots:
                path = render_finding(finding, eda_results, store, artifacts, idx)
            if path:
                rendered_count += 1
                topic = finding.get("topic", "")
                plots_meta.append({
                    "title": topic, "path": path,
                    "table": finding.get("table"), "analysis": finding.get("analysis"),
                })
                try:
                    dispatch_custom_event(
                        "eda_plot_saved",
                        {"idx": idx, "topic": topic, "path": path},
                    )
                except Exception:  # noqa: BLE001
                    pass
            rendered_findings.append({**finding, "plot_path": path})

        return {"eda_findings": rendered_findings, "eda_plots": plots_meta}

    return eda_render


# ── gate nodes (reuse the generic review gate) ───────────────────────────────

eda_review_node = make_review_node(
    lambda state: {
        "kind": "eda",
        "target_column": state.get("target_column"),
        "target_table": state.get("target_table"),
        "eda_summary": state.get("eda_summary", ""),
        "eda_findings": state.get("eda_findings", []),
        "eda_plots": state.get("eda_plots", []),
    }
)


def _revise_context(state: AgentState) -> str:
    return (
        f"Target (fixed): {state.get('target_column') or '(none)'} "
        f"in {state.get('target_table') or '(none)'}\n\n"
        f"Current analyses:\n{state.get('eda_plan', [])}\n\n"
        f"Profiles:\n{_profiles_brief(state.get('profiles', {}))}"
    )


def _revise_apply(state: AgentState, result: EdaRevision) -> dict:
    # Target is fixed by the target-definition phase; only analyses may change.
    plan = _validate_analyses(state, result.analyses) if result.analyses else state.get("eda_plan", [])
    return {"eda_plan": plan}


def make_eda_revise_node(model: BaseChatModel):
    return make_revise_node(
        model,
        system_prompt=_REVISE_SYSTEM,
        schema_model=EdaRevision,
        build_context=_revise_context,
        apply=_revise_apply,
    )
