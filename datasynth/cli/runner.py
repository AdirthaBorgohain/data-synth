"""Async graph runner: drives the LangGraph pipeline and feeds the Renderer.

Uses app.astream_events(version="v2") to capture:
  on_chain_start       → phase label (when a phase node begins)
  on_chat_model_stream → thinking blocks only (not text/tool-use)
  on_chat_model_end    → flush the streaming line
  on_chain_end         → phase results (reasoning text, selected tables, profiles)

Streaming design:
  - Anthropic extended thinking: thinking blocks stream live; text (JSON answer) is
    collected inside the node and parsed — never displayed raw.
  - No extended thinking / OpenAI: no streaming tokens appear; the reasoning text from
    the node's parsed result is shown in full after on_chain_end.
"""

from __future__ import annotations

import time

from datasynth._warnings import configure_warnings

configure_warnings()

from langchain_core.messages import AIMessageChunk
from langgraph.types import Command

from pathlib import Path

from datasynth.artifacts.store import ArtifactStore
from datasynth.cli.renderer import Renderer
from datasynth.clean.store import FrameStore
from datasynth.config import Settings
from datasynth.datasource.base import DataSource
from datasynth.graph import build_graph

_PHASE_NODES = {
    "connect", "profile", "revise", "clean_propose", "clean_apply",
    "target_propose", "target_revise", "target_apply",
    "eda_propose", "eda_run", "eda_synthesize", "eda_render", "eda_revise",
    "feat_plan", "feat_build", "feat_evaluate", "feat_critique", "feat_revise",
}

_PHASE_LABELS: dict[str, str] = {
    "connect": "Selecting relevant tables…",
    "profile": "Profiling selected tables…",
    "revise": "Revising table selection…",
    "clean_propose": "Proposing data cleaning…",
    "clean_apply": "Applying cleaning…",
    "target_propose": "Defining the prediction target…",
    "target_revise": "Revising target definition…",
    "target_apply": "Materializing target column…",
    "eda_propose": "Planning exploratory analysis…",
    "eda_run": "Running analyses and generating plots…",
    "eda_synthesize": "Synthesizing findings…",
    "eda_render": "Rendering plots for findings…",
    "eda_revise": "Revising EDA plan…",
    "feat_plan": "Designing features…",
    "feat_build": "Building feature matrix…",
    "feat_evaluate": "Scoring features with a baseline model…",
    "feat_critique": "Critiquing and refining features…",
    "feat_revise": "Revising features…",
}


def _thinking_from_chunk(chunk: AIMessageChunk) -> str:
    """Return only thinking-block text from an LLM stream chunk.

    Anthropic extended thinking arrives as content blocks:
      {"type": "thinking", "thinking": "..."}  ← show this
      {"type": "text", "text": "..."}           ← skip (JSON answer, parsed in node)
      {"type": "tool_use", ...}                 ← skip (never used now)
    Plain string content (OpenAI) → skip (it's the JSON answer text).
    """
    content = chunk.content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "thinking":
            text = block.get("thinking", "")
            if text:
                parts.append(text)
    return "".join(parts)


async def run(
    ask: str,
    run_id: str,
    settings: Settings,
    ds: DataSource,
    renderer: Renderer,
    *,
    checkpointer=None,
    frame_store: FrameStore | None = None,
    is_resume: bool = False,
) -> None:
    app = build_graph(settings, checkpointer=checkpointer)
    artifacts = ArtifactStore(settings.artifacts_dir, run_id)
    if frame_store is None:
        frame_store = FrameStore(persist_dir=Path(artifacts.dir("frames")))
    config: dict = {
        "configurable": {
            "thread_id": run_id,
            "datasource": ds,
            # Holds the working copy of each table's sample so the clean phase can
            # transform frames and re-profiling sees the result. Never serialized.
            "frame_store": frame_store,
            # Run-scoped output directory for EDA plots (and later artifacts).
            "artifacts": artifacts,
        }
    }
    initial: dict = {
        "ask": ask,
        "messages": [],
        "schema_tables": [],
        "selected_tables": [],
        "selection_reasoning": "",
        "profiles": {},
        "phase": "connect",
        "profile_issues": [],
        "clean_plan": [],
        "cleaning_log": [],
        "clean_attempted": False,
        "clean_request": "",
        "target_table": "",
        "target_column": "",
        "target_mode": "",
        "target_derivation_sql": "",
        "target_reasoning": "",
        "target_is_binary": False,
        "target_request": "",
        "needs_business_rule": False,
        "eda_plan": [],
        "eda_reasoning": "",
        "eda_results": {},
        "eda_summary": "",
        "eda_findings": [],
        "eda_plots": [],
        "entity_table": "",
        "entity_key": "",
        "feature_plan": [],
        "feat_reasoning": "",
        "feature_matrix_meta": {},
        "feature_scores": {},
        "baseline_metric": {},
        "feat_eval_summary": "",
        "feat_history": [],
        "feat_changes": {},
        "feat_action": "",
        "feat_iteration": 0,
        "feat_build_log": [],
    }

    t0 = time.monotonic()
    shown_log = 0  # how many cleaning_log entries already rendered
    last_eda_target = ""  # target from propose/revise for tool_result display

    # Resume loop: each astream_events pass runs until the graph finishes or hits
    # an interrupt() (the review gate). On interrupt we collect a comment and
    # resume with Command(resume=...); a blank comment lets the graph proceed.
    input_data: object = None if is_resume else initial
    while True:
        async for event in app.astream_events(input_data, config, version="v2"):
            kind: str = event["event"]
            name: str = event.get("name", "")
            node: str = event.get("metadata", {}).get("langgraph_node", "")

            # ── phase start ──────────────────────────────────────────────
            if kind == "on_chain_start" and node in _PHASE_NODES and name == node:
                renderer.phase_start(node, _PHASE_LABELS.get(node, ""))

            # ── LLM thinking stream (thinking blocks only) ───────────────
            elif kind == "on_chat_model_stream":
                text = _thinking_from_chunk(event["data"]["chunk"])
                if text:
                    renderer.stream_token(text)

            # ── LLM call done → flush the streaming line ─────────────────
            elif kind == "on_chat_model_end":
                renderer.stream_end()

            # ── live ticks from inside nodes (eda_run / eda_synthesize) ──
            elif kind == "on_custom_event":
                ename = event.get("name", "")
                data = event.get("data") or {}
                if ename == "eda_analysis":
                    renderer.eda_analysis_tick(data)
                elif ename == "eda_summary":
                    renderer.eda_summary_tick(data.get("text", ""))
                elif ename == "eda_finding":
                    renderer.eda_finding_tick(data)
                elif ename == "eda_plot_saved":
                    renderer.eda_plot_saved_tick(data)

            # ── phase results ─────────────────────────────────────────────
            elif kind == "on_chain_end" and node in _PHASE_NODES and name == node:
                output: dict = event["data"].get("output") or {}

                if node == "connect":
                    reasoning: str = output.get("selection_reasoning", "")
                    tables: list[str] = output.get("selected_tables", [])
                    # Always show reasoning — it's the LLM's stated explanation.
                    # With extended thinking: thinking already streamed live + this is the summary.
                    # Without thinking (OpenAI, llm_reasoning=False): this is the only visible rationale.
                    if reasoning:
                        renderer.narration(reasoning)
                    renderer.tool_result("Selected", tables)

                elif node == "profile":
                    for table, profile in (output.get("profiles") or {}).items():
                        renderer.profile_table(table, profile)

                elif node == "revise":
                    reasoning = output.get("selection_reasoning", "")
                    if reasoning:
                        renderer.narration(reasoning)
                    if output.get("revise_action") == "revise":
                        renderer.tool_result("Revised", output.get("selected_tables", []))

                elif node == "clean_apply":
                    log = output.get("cleaning_log", [])
                    renderer.clean_summary(log[shown_log:])
                    shown_log = len(log)

                elif node == "target_apply":
                    mode = output.get("target_mode")
                    col = output.get("target_column", "")
                    if mode == "unsupervised" or not col:
                        renderer.tool_result("Target", ["(unsupervised — no label)"])
                    else:
                        renderer.tool_result("Target ready", [col])

                elif node in ("eda_propose", "eda_revise"):
                    reasoning = output.get("eda_reasoning", "")
                    if reasoning:
                        renderer.narration(reasoning)
                    last_eda_target = output.get("target_column") or ""
                    if last_eda_target:
                        renderer.tool_result("Target", [last_eda_target])

                elif node == "feat_plan":
                    reasoning = output.get("feat_reasoning", "")
                    if reasoning:
                        renderer.narration(reasoning)
                    renderer.feat_plan_summary(output.get("feature_plan", []))

                elif node == "feat_build":
                    renderer.feat_build_summary(
                        output.get("feature_matrix_meta", {}),
                        output.get("feat_build_log", []),
                    )

                elif node == "feat_evaluate":
                    renderer.feat_scorecard(
                        output.get("feature_scores", {}),
                        output.get("baseline_metric", {}),
                    )

                elif node == "feat_critique":
                    reasoning = output.get("feat_reasoning", "")
                    if reasoning:
                        renderer.narration(reasoning)
                    renderer.feat_critique_summary(
                        output.get("feat_changes", {}),
                        output.get("feat_history", []),
                        output.get("feat_action", ""),
                    )

                elif node == "feat_revise":
                    reasoning = output.get("feat_reasoning", "")
                    if reasoning:
                        renderer.narration(reasoning)

                renderer.mark_stage_done(node)

        # Graph paused on an interrupt iff there is a next node to run.
        snapshot = await app.aget_state(config)
        if not snapshot.next:
            break
        payload = snapshot.tasks[0].interrupts[0].value
        kind = payload.get("kind")
        if kind == "clean":
            comment = renderer.clean_review_prompt(
                payload.get("clean_plan", []), payload.get("clean_reasoning", "")
            )
        elif kind == "target":
            comment = renderer.target_review_prompt(
                payload.get("mode", ""),
                payload.get("target_table", ""),
                payload.get("target_column", ""),
                payload.get("derivation_sql", ""),
                payload.get("reasoning", ""),
                bool(payload.get("needs_business_rule", False)),
            )
        elif kind == "eda":
            comment = renderer.eda_review_prompt(
                payload.get("target_column"),
                payload.get("eda_summary", ""),
                payload.get("eda_findings", []),
                payload.get("eda_plots", []),
            )
        elif kind == "features":
            comment = renderer.feat_review_prompt(
                payload.get("feature_matrix_meta", {}),
                payload.get("feature_scores", {}),
                payload.get("baseline_metric", {}),
                payload.get("feat_eval_summary", ""),
                payload.get("feat_history", []),
            )
        else:
            comment = renderer.review_prompt(payload.get("selected_tables", []))
        input_data = Command(resume=comment)

    renderer.run_footer(
        time.monotonic() - t0,
        "Knowledge graph → Model training",
    )
