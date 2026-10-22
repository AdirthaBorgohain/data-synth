"""Compile the datasynth LangGraph pipeline.

Implemented: connect → profile → [clean] → table review → EDA → feature-gen (all with
review interrupts where noted in graph wiring).

Planned after feature review: kg → model [interrupt] → explain → report
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from datasynth.config import Settings
from datasynth.graph.nodes.clean import (
    clean_apply_node,
    clean_review_node,
    make_clean_propose_node,
    make_clean_revise_node,
)
from datasynth.graph.nodes.connect import make_connect_node
from datasynth.graph.nodes.eda import (
    eda_review_node,
    make_eda_propose_node,
    make_eda_render_node,
    make_eda_revise_node,
    make_eda_run_node,
    make_eda_synthesize_node,
)
from datasynth.graph.nodes.features import (
    feat_review_node,
    make_feat_build_node,
    make_feat_critique_node,
    make_feat_evaluate_node,
    make_feat_plan_node,
    make_feat_revise_node,
    route_after_critique,
)
from datasynth.graph.nodes.profile import profile_node
from datasynth.graph.nodes.select_review import (
    make_select_revise_node,
    select_review_node,
)
from datasynth.graph.nodes.target import (
    make_target_apply_node,
    make_target_propose_node,
    make_target_revise_node,
    target_review_node,
)
from datasynth.graph.review import make_route_after_revise, route_after_review
from datasynth.graph.state import AgentState
from datasynth.llm.provider import get_chat_model


def route_after_profile(state: AgentState) -> str:
    """Enter clean when profiling flagged issues and clean hasn't run yet."""
    if state.get("profile_issues") and not state.get("clean_attempted"):
        return "clean"
    return "review"


def build_graph(settings: Settings, checkpointer: BaseCheckpointSaver | None = None):
    """Return a compiled LangGraph app ready to invoke.

    Pass a SqliteSaver (or any BaseCheckpointSaver) to persist checkpoints across
    process restarts and enable --resume. Defaults to MemorySaver (in-memory only).

    Usage:
        app = build_graph(settings)
        config = {"configurable": {"thread_id": run_id, "datasource": ds}}
        state = app.invoke({"ask": "...", "phase": "connect"}, config=config)
    """
    model = get_chat_model(settings)
    eda_cfg = {
        "top_k": settings.eda_top_k,
        "corr_threshold": settings.eda_corr_threshold,
        "max_plots": settings.eda_max_plots,
    }
    feat_cfg = {
        "sample_size": settings.sample_size,
        "max_rows": settings.max_result_rows,
        "max_iterations": settings.feat_max_iterations,
        "epsilon": settings.feat_metric_epsilon,
    }

    builder = StateGraph(AgentState)
    builder.add_node("connect", make_connect_node(model))
    builder.add_node("profile", profile_node)
    builder.add_node("review", select_review_node)
    builder.add_node("revise", make_select_revise_node(model))
    builder.add_node("clean_propose", make_clean_propose_node(model))
    builder.add_node("clean_review", clean_review_node)
    builder.add_node("clean_revise", make_clean_revise_node(model))
    builder.add_node("clean_apply", clean_apply_node)
    builder.add_node("target_propose", make_target_propose_node(model))
    builder.add_node("target_review", target_review_node)
    builder.add_node("target_revise", make_target_revise_node(model))
    builder.add_node("target_apply", make_target_apply_node(feat_cfg))
    builder.add_node("eda_propose", make_eda_propose_node(model))
    builder.add_node("eda_run", make_eda_run_node(eda_cfg))
    builder.add_node("eda_synthesize", make_eda_synthesize_node(model))
    builder.add_node("eda_render", make_eda_render_node(eda_cfg))
    builder.add_node("eda_review", eda_review_node)
    builder.add_node("eda_revise", make_eda_revise_node(model))
    builder.add_node("feat_plan", make_feat_plan_node(model, feat_cfg))
    builder.add_node("feat_build", make_feat_build_node(feat_cfg))
    builder.add_node("feat_evaluate", make_feat_evaluate_node(feat_cfg))
    builder.add_node("feat_critique", make_feat_critique_node(model))
    builder.add_node("feat_review", feat_review_node)
    builder.add_node("feat_revise", make_feat_revise_node(model))

    builder.add_edge(START, "connect")
    builder.add_edge("connect", "profile")
    # After profiling: clean if issues flagged (once), else go to table review.
    builder.add_conditional_edges(
        "profile", route_after_profile,
        {"clean": "clean_propose", "review": "review"},
    )

    # ── table-selection review (proceed → target definition) ──
    builder.add_conditional_edges(
        "review", route_after_review,
        {"revise": "revise", "proceed": "target_propose"},
    )
    # revise: change tables → re-profile; clean request → clean phase; else → target.
    builder.add_conditional_edges(
        "revise",
        make_route_after_revise(
            {"proceed": "target_propose", "revise": "profile", "clean": "clean_propose"}
        ),
        {"target_propose": "target_propose", "profile": "profile", "clean_propose": "clean_propose"},
    )

    # ── target-definition gate (approval-gated, reuses the review gate) ──
    builder.add_edge("target_propose", "target_review")
    builder.add_conditional_edges(
        "target_review", route_after_review,
        {"proceed": "target_apply", "revise": "target_revise"},
    )
    builder.add_conditional_edges(
        "target_revise",
        make_route_after_revise(
            {"proceed": "target_apply", "revise": "target_review"}
        ),
        {"target_apply": "target_apply", "target_review": "target_review"},
    )
    builder.add_edge("target_apply", "eda_propose")

    # ── data-cleaning loop (approval-gated, reuses the review gate) ──
    builder.add_edge("clean_propose", "clean_review")
    # Blank comment approves → apply; a comment routes to clean_revise.
    builder.add_conditional_edges(
        "clean_review", route_after_review,
        {"proceed": "clean_apply", "revise": "clean_revise"},
    )
    # clean_revise: adjusted plan → re-show; "go" → apply.
    builder.add_conditional_edges(
        "clean_revise",
        make_route_after_revise(
            {"proceed": "clean_apply", "revise": "clean_review"}
        ),
        {"clean_apply": "clean_apply", "clean_review": "clean_review"},
    )
    # Apply cleaning, then re-profile the cleaned data.
    builder.add_edge("clean_apply", "profile")

    # ── EDA: propose → run → synthesize → review (approval-gated) ──
    builder.add_edge("eda_propose", "eda_run")
    builder.add_edge("eda_run", "eda_synthesize")
    builder.add_edge("eda_synthesize", "eda_render")
    builder.add_edge("eda_render", "eda_review")
    # Blank comment proceeds to feature gen; a comment routes to eda_revise.
    builder.add_conditional_edges(
        "eda_review", route_after_review,
        {"proceed": "feat_plan", "revise": "eda_revise"},
    )
    # eda_revise: changed target/analyses → re-run EDA; approval → feature gen.
    builder.add_conditional_edges(
        "eda_revise",
        make_route_after_revise({"proceed": "feat_plan", "revise": "eda_run"}),
        {"feat_plan": "feat_plan", "eda_run": "eda_run"},
    )

    # ── feature-gen agentic loop: plan → build → evaluate → critique ──
    builder.add_edge("feat_plan", "feat_build")
    builder.add_edge("feat_build", "feat_evaluate")
    builder.add_edge("feat_evaluate", "feat_critique")
    # critique: keep refining (loop to build) while improving + under the cap, else review.
    builder.add_conditional_edges(
        "feat_critique", route_after_critique,
        {"refine": "feat_build", "review": "feat_review"},
    )
    # feat_review: blank comment finishes; a comment routes to feat_revise.
    builder.add_conditional_edges(
        "feat_review", route_after_review,
        {"proceed": END, "revise": "feat_revise"},
    )
    # feat_revise: changed specs → rebuild + re-evaluate; approval → END.
    builder.add_conditional_edges(
        "feat_revise",
        make_route_after_revise({"proceed": END, "revise": "feat_build"}),
        {END: END, "feat_build": "feat_build"},
    )

    cp = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=cp)