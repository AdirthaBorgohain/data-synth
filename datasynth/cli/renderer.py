"""Terminal renderer — owns all output. No caller should use print() directly.

Visual design: Catppuccin Mocha palette, consistent phase identity, Unicode
null/importance bars, plotext inline charts, and prompt_toolkit review gates.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from datasynth import __version__
from datasynth.cli.theme import (
    GREEN, LAVENDER, MAUVE, PEACH, RED, SAPPHIRE, SKY, SUBTEXT, TEAL, TEXT, YELLOW,
    NULL_BAR_EMPTY, NULL_BAR_FULL, NULL_BAR_WIDTH, NULL_HIGH,
    PIPELINE_STAGES,
    PLOT_COLOR_1, PLOT_COLOR_2, PLOT_HEIGHT, PLOT_WIDTH,
    PROMPT_STYLE_PAIRS,
    S_PANEL_HEADER, S_PANEL_REVIEW, S_TABLE_HEADER,
    BANNER_GRADIENT, BANNER_LOGO, BANNER_LOGO_GAP, gradient_hex,
)

_NODE_TO_STAGE: dict[str, str] = {
    "connect":        "connect",
    "profile":        "profile",
    "revise":         "profile",
    "clean_propose":  "clean",
    "clean_review":   "clean",
    "clean_revise":   "clean",
    "clean_apply":    "clean",
    "target_propose": "target",
    "target_review":  "target",
    "target_revise":  "target",
    "target_apply":   "target",
    "eda_propose":    "eda",
    "eda_run":        "eda",
    "eda_synthesize": "eda",
    "eda_render":     "eda",
    "eda_review":     "eda",
    "eda_revise":     "eda",
    "feat_plan":      "features",
    "feat_build":     "features",
    "feat_evaluate":  "features",
    "feat_critique":  "features",
    "feat_review":    "features",
    "feat_revise":    "features",
}


def _target_mode_summary(mode: str, target_table: str, target_column: str) -> str:
    label = (
        f"{target_table}.{target_column}"
        if target_table and target_column
        else (target_column or "")
    )
    if mode == "unsupervised":
        return (
            "No label column — this is an unsupervised task "
            "(patterns or groups, not predicting a specific column)."
        )
    if mode == "derived":
        if label:
            return f"The label will be a new column, {label}, computed from the SQL rule below."
        return "The label will be a new column computed from the SQL rule below."
    if mode == "existing":
        if label:
            return f"The label is the existing column {label} in your database."
        return "The label is an existing column in your database (not chosen yet)."
    return f"Label setup: {mode or 'unknown'}."


def _target_suggestion_line(mode: str, target_table: str, target_column: str) -> str:
    """One-line TL;DR recap shown at the bottom of the target review panel."""
    label = (
        f"{target_table}.{target_column}"
        if target_table and target_column
        else (target_column or "")
    )
    if mode == "unsupervised":
        return "Suggestion: treat as unsupervised — no target column."
    if mode == "derived":
        return f"Suggestion: derive new target {label or 'column'} from the SQL rule above."
    if mode == "existing":
        return f"Suggestion: use existing column {label or '(unknown)'} as the target."
    return f"Suggestion: {mode or 'unknown'}."


class Renderer:
    def __init__(self) -> None:
        self._console = Console(highlight=False)
        self._streaming = False
        self._stage_status: dict[str, str] = {key: "pending" for key, _ in PIPELINE_STAGES}
        self._pt_history = None  # InMemoryHistory, lazy-init on first _pt_input call
        self._live = None          # rich.Live animating the active stage glyph
        self._tracker_thread = None
        self._tracker_stop = None

    # ── private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _null_bar(rate: float) -> str:
        filled = round(rate * NULL_BAR_WIDTH)
        if rate < 0.20:
            color = GREEN
        elif rate < NULL_HIGH:
            color = TEAL
        elif rate < 0.70:
            color = PEACH
        else:
            color = RED
        bar = NULL_BAR_FULL * filled + NULL_BAR_EMPTY * (NULL_BAR_WIDTH - filled)
        return f"[{color}]{bar}  {rate:.0%}[/{color}]"

    @staticmethod
    def _importance_bar(value: float, max_val: float, width: int = 12) -> str:
        if max_val <= 0:
            return ""
        filled = round((value / max_val) * width)
        return f"[{SAPPHIRE}]{'█' * filled}[/{SAPPHIRE}]"

    def _advance_stage(self, node: str) -> None:
        new_stage = _NODE_TO_STAGE.get(node)
        if not new_stage:
            return
        for key in self._stage_status:
            if self._stage_status[key] == "active":
                self._stage_status[key] = "done"
        self._stage_status[new_stage] = "active"

    _ACTIVE_FRAMES = ["◐", "◓", "◑", "◒"]

    def _render_tracker(self, frame: int | None = None) -> str:
        parts = []
        for key, label in PIPELINE_STAGES:
            if key == "clean" and self._stage_status.get("clean") == "pending":
                continue
            status = self._stage_status.get(key, "pending")
            if status == "done":
                glyph = f"[{GREEN}]✓[/{GREEN}]"
                label_style = f"dim {SUBTEXT}"
            elif status == "active":
                ch = "◉" if frame is None else self._ACTIVE_FRAMES[frame % len(self._ACTIVE_FRAMES)]
                glyph = f"[{MAUVE}]{ch}[/{MAUVE}]"
                label_style = f"bold {MAUVE}"
            else:
                glyph = f"[{SUBTEXT}]○[/{SUBTEXT}]"
                label_style = f"dim {SUBTEXT}"
            parts.append(f"{glyph} [{label_style}]{label}[/{label_style}]")
        sep = f"  [{SUBTEXT}]·[/{SUBTEXT}]  "
        return "  " + sep.join(parts)

    def _print_stage_tracker(self) -> None:
        self._console.print(self._render_tracker(None))

    # ── animated stage tracker (pulses the active glyph while a phase runs) ──

    def _start_tracker_animation(self) -> None:
        self._stop_tracker_animation()
        if not self._console.is_terminal:
            self._print_stage_tracker()
            return
        import threading

        from rich.live import Live
        from rich.text import Text

        self._live = Live(
            Text.from_markup(self._render_tracker(0)),
            console=self._console,
            refresh_per_second=12,
            transient=False,
        )
        self._live.start()
        self._tracker_stop = threading.Event()

        def _loop() -> None:
            import time
            i = 0
            while not self._tracker_stop.is_set():
                i += 1
                try:
                    self._live.update(Text.from_markup(self._render_tracker(i)))
                except Exception:
                    break
                time.sleep(0.12)

        self._tracker_thread = threading.Thread(target=_loop, daemon=True)
        self._tracker_thread.start()

    def _stop_tracker_animation(self) -> None:
        if self._live is None:
            return
        if self._tracker_stop is not None:
            self._tracker_stop.set()
        if self._tracker_thread is not None:
            self._tracker_thread.join(timeout=0.3)
        try:
            from rich.text import Text
            self._live.update(Text.from_markup(self._render_tracker(None)))  # freeze clean ◉
            self._live.stop()
        finally:
            self._live = None
            self._tracker_thread = None
            self._tracker_stop = None

    def _pt_input(self, prompt_text: str, hint: str = "", validator=None) -> str:
        import asyncio
        import concurrent.futures

        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.styles import Style

        if self._pt_history is None:
            from prompt_toolkit.history import InMemoryHistory
            self._pt_history = InMemoryHistory()

        style = Style.from_dict(dict(PROMPT_STYLE_PAIRS))
        bottom_toolbar = (
            HTML(f'<bottom-toolbar>  {hint}  </bottom-toolbar>') if hint else None
        )
        prompt_str = HTML(
            f'<prompt>  {prompt_text} </prompt>'
            f'<hint>&#8250;</hint> '
        )

        def _run_prompt() -> str:
            try:
                return pt_prompt(
                    prompt_str,
                    style=style,
                    bottom_toolbar=bottom_toolbar,
                    history=self._pt_history,
                    validator=validator,
                    validate_while_typing=False,
                ).strip()
            except (EOFError, KeyboardInterrupt):
                return ""

        try:
            asyncio.get_running_loop()
            # Inside a running asyncio loop — run in a worker thread to avoid
            # nested event loop error from prompt_toolkit.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(_run_prompt).result()
        except RuntimeError:
            return _run_prompt()

    # ── welcome screen ────────────────────────────────────────────────────

    def _gradient_line(self, line: str, width: int, row: int, rows: int) -> Text:
        """Color one composite banner row with a diagonal left→right sweep.

        The per-row offset tilts the gradient so color rakes across the
        wordmark and logo at an angle, like light over the glyphs.
        """
        t = Text("  ")
        skew = (row / max(rows, 1)) * 0.30  # diagonal tilt
        for col, ch in enumerate(line):
            if ch == " ":
                t.append(" ")
                continue
            pos = (col / max(width, 1) + skew) % 1.0
            color = gradient_hex(BANNER_GRADIENT, pos)
            t.append(ch, style=f"bold {color}")
        return t

    @staticmethod
    def _compose_banner(
        wordmark: list[str],
        logo: list[str],
        gap: int,
        *,
        version: str = "",
    ) -> list[str]:
        """Place the logo to the right of the wordmark, vertically centered."""
        version_suffix = f"  v{version}" if version else ""
        wm_width = max((len(ln) for ln in wordmark), default=0)
        if version_suffix and wordmark:
            last = len(wordmark) - 1
            wm_width = max(wm_width, len(wordmark[last]) + len(version_suffix))
        rows = max(len(wordmark), len(logo))
        wm_top = (rows - len(wordmark)) // 2
        logo_top = (rows - len(logo)) // 2
        version_row = wm_top + len(wordmark) - 1 if version_suffix else -1
        out: list[str] = []
        for r in range(rows):
            left = wordmark[r - wm_top] if wm_top <= r < wm_top + len(wordmark) else ""
            if r == version_row:
                left = left + version_suffix
            left = left.ljust(wm_width)
            right = logo[r - logo_top] if logo_top <= r < logo_top + len(logo) else ""
            out.append(left + (" " * gap) + right)
        return out

    def _print_data_source_status(self, connected: bool, db_name: str) -> None:
        line = Text("  ")
        if connected:
            line.append("● ", style=GREEN)
            line.append("Connected to PostgreSQL", style=TEXT)
            if db_name:
                line.append(" [", style=f"dim {SUBTEXT}")
                line.append(db_name, style=TEAL)
                line.append("]", style=f"dim {SUBTEXT}")
        else:
            line.append("● ", style=RED)
            line.append("Data source not connected", style=RED)
            if db_name and db_name != "unknown":
                line.append(" [", style=f"dim {SUBTEXT}")
                line.append(db_name, style=f"dim {SUBTEXT}")
                line.append("]", style=f"dim {SUBTEXT}")
        self._console.print(line)

    def welcome(
        self,
        animate: bool = True,
        *,
        data_source_connected: bool | None = None,
        data_source_db: str = "",
    ) -> None:
        try:
            import pyfiglet
            art = pyfiglet.figlet_format("datasynth", font="slant")
            wordmark = [ln for ln in art.splitlines() if ln.strip()]
        except ImportError:
            wordmark = ["datasynth"]

        lines = self._compose_banner(
            wordmark,
            BANNER_LOGO,
            BANNER_LOGO_GAP,
            version=__version__,
        )

        animate = animate and self._console.is_terminal
        delay = 0.04 if animate else 0.0

        width = max((len(ln) for ln in lines), default=1)
        self._console.print()
        self._console.print(Rule(style=f"dim {SUBTEXT}"))
        self._console.print()
        for row, line in enumerate(lines):
            self._console.print(self._gradient_line(line, width, row, len(lines)))
            if delay:
                import time
                time.sleep(delay)
        self._console.print()

        tagline = Text("  ")
        tagline.append("◆ ", style=SAPPHIRE)
        tagline.append("Agentic ML Pipeline", style=f"bold {LAVENDER}")
        tagline.append("  ·  ", style=f"dim {SUBTEXT}")
        for i, stage in enumerate(["Connect", "Profile", "Analyze", "Model"]):
            if i:
                tagline.append(" → ", style=f"dim {SUBTEXT}")
            color = gradient_hex(BANNER_GRADIENT, i / 3)
            tagline.append(stage, style=color)
        self._console.print(tagline)
        if data_source_connected is not None:
            self._print_data_source_status(
                connected=data_source_connected,
                db_name=data_source_db,
            )
        self._console.print()
        self._console.print(Rule(style=f"dim {SUBTEXT}"))
        self._console.print()

    # ── run lifecycle ──────────────────────────────────────────────────────

    def run_header(self, run_id: str, ask: str, provider: str) -> None:
        header = Text()
        header.append("  datasynth", style=f"bold {TEXT}")
        header.append("  ·  ", style=f"dim {SUBTEXT}")
        header.append(run_id, style=MAUVE)
        header.append("  ·  ", style=f"dim {SUBTEXT}")
        header.append(provider, style=SAPPHIRE)
        header.append(f'\n\n  "{ask}"', style=f"italic {TEXT}")
        self._console.print()
        self._console.print(
            Panel(header, border_style=S_PANEL_HEADER, box=box.ROUNDED, padding=(1, 2))
        )
        self._console.print()
        self._print_stage_tracker()
        self._console.print()

    def run_footer(self, duration_s: float, next_phases: str) -> None:
        self._stop_tracker_animation()
        for key in self._stage_status:
            if self._stage_status[key] == "active":
                self._stage_status[key] = "done"
        self._console.print()
        self._print_stage_tracker()
        self._console.print()
        self._console.print(Rule(style=MAUVE))
        t = Text("  ")
        t.append("Pipeline complete", style=f"bold {GREEN}")
        t.append(f"  {duration_s:.1f}s", style=SAPPHIRE)
        if next_phases:
            t.append(f"  ·  Next: {next_phases}", style=f"dim {SUBTEXT}")
        self._console.print(t)
        self._console.print()

    def mark_stage_done(self, node: str) -> None:
        stage = _NODE_TO_STAGE.get(node)
        if stage and self._stage_status.get(stage) == "active":
            self._stage_status[stage] = "done"

    # ── phase labels ───────────────────────────────────────────────────────

    def phase_start(self, phase: str, detail: str = "") -> None:
        self._advance_stage(phase)
        t = Text("\n")
        t.append("→", style=f"bold {MAUVE}")
        t.append(f"  [{phase}]", style=f"dim {MAUVE}")
        if detail:
            t.append(f"  {detail}", style=TEXT)
        self._console.print(t)
        self._start_tracker_animation()

    # ── LLM token streaming ────────────────────────────────────────────────

    def stream_token(self, token: str) -> None:
        if not token:
            return
        if not self._streaming:
            self._stop_tracker_animation()
            self._console.print(f"  [{SUBTEXT}]thinking:[/{SUBTEXT}] ", end="")
            self._streaming = True
        self._console.out(token, end="", highlight=False, style=LAVENDER)

    def stream_end(self) -> None:
        self._stop_tracker_animation()
        if self._streaming:
            self._console.print()
            self._console.print(Rule(style=f"dim {SUBTEXT}"))
            self._streaming = False

    # ── results ───────────────────────────────────────────────────────────

    def tool_result(self, label: str, items: list[str]) -> None:
        self._stop_tracker_animation()
        value = ", ".join(items) if items else "(none)"
        t = Text("  ")
        t.append("✓", style=f"bold {GREEN}")
        t.append(f"  {label}:", style=f"dim {GREEN}")
        t.append(f" {value}", style=TEXT)
        self._console.print(t)

    def profile_table(self, table: str, profile: dict) -> None:
        self._stop_tracker_animation()
        if "error" in profile:
            t = Text("\n  ")
            t.append("✗", style=f"bold {RED}")
            t.append(f"  {table}: {profile['error']}", style=TEXT)
            self._console.print(t)
            return

        n = profile.get("row_count_sample", "?")
        n_str = f"{n:,}" if isinstance(n, int) else str(n)
        heading = Text("\n  ")
        heading.append(table, style=f"bold {MAUVE}")
        heading.append(f"  ({n_str} sample rows)", style=f"dim {SUBTEXT}")
        self._console.print(heading)

        cols = profile.get("columns", {})
        if not cols:
            self._console.print(f"  [{SUBTEXT}](no columns)[/{SUBTEXT}]")
            return

        tbl = Table(box=box.SIMPLE_HEAD, padding=(0, 1), show_edge=False, show_header=True)
        tbl.add_column("column",   style=TEXT,             no_wrap=True, header_style=S_TABLE_HEADER)
        tbl.add_column("dtype",    style=f"dim {SUBTEXT}", no_wrap=True, header_style=S_TABLE_HEADER)
        tbl.add_column("null values",                      no_wrap=True, header_style=S_TABLE_HEADER)
        tbl.add_column("n_unique", justify="right",        no_wrap=True, header_style=S_TABLE_HEADER,
                       style=f"dim {SUBTEXT}")

        for col_name, stats in cols.items():
            null_rate = stats.get("null_rate", 0.0)
            n_unique = stats.get("n_unique")
            unique_str = f"{n_unique:,}" if isinstance(n_unique, int) else "—"
            tbl.add_row(
                col_name,
                stats.get("dtype", "?"),
                self._null_bar(null_rate),
                unique_str,
            )

        self._console.print("  ", end="")
        self._console.print(tbl)

    # ── non-streamed narration ────────────────────────────────────────────

    def narration(self, text: str) -> None:
        self.stream_end()
        for i, para in enumerate(text.split("\n\n")):
            if i > 0:
                self._console.print()
            indented = "  " + para.replace("\n", "\n  ")
            self._console.out(indented, highlight=False, style=TEXT)

    # ── approval gate ─────────────────────────────────────────────────────

    def approval_prompt(self, proposal: str) -> bool:
        self.stream_end()
        self._console.print(
            Panel(proposal, title="Proposed operation", border_style=S_PANEL_REVIEW, padding=(0, 1))
        )
        from prompt_toolkit.validation import Validator
        validator = Validator.from_callable(
            lambda t: t.strip().lower() in ("y", "n", "yes", "no", ""),
            error_message="Type y or n",
            move_cursor_to_end=True,
        )
        answer = self._pt_input(
            "Approve? [y/n]",
            hint="y = approve  ·  n = reject  ·  Ctrl-C = abort",
            validator=validator,
        )
        return answer.lower() in ("y", "yes")

    # ── review gate (post-profile table-selection comment) ──────────────────

    def review_prompt(self, selected_tables: list[str]) -> str:
        self.stream_end()
        body = Text()
        tables = ", ".join(selected_tables) if selected_tables else "(none)"
        body.append("Selected tables: ", style=f"bold {TEXT}")
        body.append(f"{tables}\n\n", style=TEXT)
        body.append(
            "Comment to revise the selection, or press Enter to continue.",
            style=f"dim {SUBTEXT}",
        )
        self._console.print(
            Panel(body, title="Review selection", border_style=S_PANEL_REVIEW, padding=(0, 1))
        )
        return self._pt_input(
            "Comment",
            hint="Enter = continue  ·  type to revise selection  ·  Ctrl-C = abort",
        )

    # ── data-cleaning gate ──────────────────────────────────────────────────

    @staticmethod
    def _format_op(op: dict) -> str:
        target = op.get("table", "")
        if op.get("column"):
            target += f".{op['column']}"
        params = op.get("params") or {}
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        head = f"{target}: {op.get('op')}" + (f"({param_str})" if param_str else "")
        reason = op.get("reason")
        return head + (f" — {reason}" if reason else "")

    def clean_review_prompt(self, plan: list[dict], reasoning: str = "") -> str:
        self.stream_end()
        body = Text()
        if reasoning:
            body.append(f"{reasoning}\n\n", style=TEXT)
        if plan:
            for op in plan:
                body.append("• ", style=TEAL)
                body.append(f"{self._format_op(op)}\n", style=TEXT)
        else:
            body.append("(no operations proposed)\n", style=f"dim {SUBTEXT}")
        body.append(
            "\nComment to adjust the plan, or press Enter to apply.",
            style=f"dim {SUBTEXT}",
        )
        self._console.print(
            Panel(body, title="Proposed cleaning", border_style=S_PANEL_REVIEW, padding=(0, 1))
        )
        return self._pt_input(
            "Comment",
            hint="Enter = apply plan  ·  type to adjust  ·  Ctrl-C = skip",
        )

    def clean_summary(self, applied: list[dict]) -> None:
        self.stream_end()
        if not applied:
            self._console.print(f"  [{SUBTEXT}]No cleaning operations applied.[/{SUBTEXT}]")
            return
        for op in applied:
            target = op.get("table", "")
            if op.get("column"):
                target += f".{op['column']}"
            if op.get("applied"):
                delta = ""
                rb, ra = op.get("rows_before"), op.get("rows_after")
                if isinstance(rb, int) and isinstance(ra, int) and ra != rb:
                    delta = f" ({rb - ra:+d} rows)"
                t = Text("  ")
                t.append("✓", style=f"bold {GREEN}")
                t.append(f"  {target}:", style=f"dim {GREEN}")
                t.append(f" {op.get('op')}{delta}", style=TEXT)
                self._console.print(t)
            else:
                t = Text("  ")
                t.append("✗", style=f"bold {RED}")
                t.append(f"  {target}: {op.get('op')} skipped — {op.get('error', '')}", style=TEXT)
                self._console.print(t)

    # ── target-definition phase ────────────────────────────────────────────

    def target_review_prompt(
        self,
        mode: str,
        target_table: str,
        target_column: str,
        derivation_sql: str,
        reasoning: str,
        needs_business_rule: bool,
    ) -> str:
        self.stream_end()
        body = Text()
        body.append(
            _target_mode_summary(mode=mode, target_table=target_table, target_column=target_column) + "\n",
            style=TEXT,
        )
        if mode == "derived" and derivation_sql:
            body.append("\nSQL rule:\n", style=f"bold {TEXT}")
            body.append(f"  {derivation_sql}\n", style=SKY)
        if reasoning:
            body.append(f"\n{reasoning}\n", style=TEXT)
        body.append(
            "\n" + _target_suggestion_line(mode=mode, target_table=target_table, target_column=target_column) + "\n",
            style=f"bold {TEAL}",
        )
        if needs_business_rule:
            body.append(
                "\nThis target encodes a business-rule choice. Reply with your rule "
                "(e.g. 'churn = no order in 60 days'), blank to accept, or 'unsupervised' to skip.",
                style=PEACH,
            )
        else:
            body.append(
                "\nComment to revise (or write a business rule), or press Enter to accept.",
                style=f"dim {SUBTEXT}",
            )
        self._console.print(
            Panel(body, title="Review target", border_style=S_PANEL_REVIEW, padding=(0, 1))
        )
        return self._pt_input(
            "Business rule or comment",
            hint="Enter = accept  ·  type to refine target  ·  Ctrl-C = abort",
        )

    # ── EDA phase ───────────────────────────────────────────────────────────

    def eda_analysis_tick(self, payload: dict) -> None:
        self.stream_end()
        table = payload.get("table", "")
        analysis = payload.get("analysis", "")
        summary = payload.get("summary", "")
        if payload.get("ok"):
            t = Text("  ")
            t.append("✓", style=f"bold {GREEN}")
            t.append(f"  {table}", style=f"dim {SUBTEXT}")
            t.append(" · ", style=f"dim {SUBTEXT}")
            t.append(analysis, style=TEXT)
            if summary:
                t.append(f": {summary}", style=f"dim {SUBTEXT}")
            self._console.print(t)
        else:
            err = payload.get("error") or summary or "failed"
            t = Text("  ")
            t.append("⚠", style=f"bold {PEACH}")
            t.append(f"  {table}", style=f"dim {SUBTEXT}")
            t.append(f" · {analysis}: {err}", style=PEACH)
            self._console.print(t)

    def eda_summary_tick(self, text: str) -> None:
        if not text:
            return
        self.stream_end()
        indented = "  " + text.replace("\n", "\n  ")
        heading = Text()
        heading.append("Findings:", style=f"bold {MAUVE}")
        self._console.print(heading)
        self._console.out(indented, highlight=False, style=TEXT)

    def eda_finding_tick(self, finding: dict) -> None:
        self.stream_end()
        topic = finding.get("topic", "")
        obs = finding.get("observation", "")
        impl = finding.get("downstream_implication", "")
        self._console.print()
        t = Text("  ")
        t.append("▸", style=f"bold {MAUVE}")
        t.append(f"  {topic}", style=f"bold {TEXT}")
        t.append(f": {obs}", style=TEXT)
        self._console.print(t)
        if impl:
            impl_t = Text("      ")
            impl_t.append(f"→ {impl}", style=GREEN)
            self._console.print(impl_t)

    def eda_plot_saved_tick(self, payload: dict) -> None:
        self.stream_end()
        topic = payload.get("topic", "")
        path = payload.get("path", "")
        if not path:
            return
        abs_path = os.path.abspath(path)
        link = f"[link=file://{abs_path}]{path}[/link]"
        self._console.print(f"  [dim {GREEN}]📁 {topic}:[/dim {GREEN}] {link}")

    def _eda_body(self, target, summary, findings, plots) -> Text:
        body = Text()
        body.append("Target: ", style=f"bold {TEXT}")
        body.append(f"{target or '(none inferred)'}\n", style=TEXT)
        if findings:
            body.append(f"Findings: {len(findings)}\n", style=TEXT)
        if plots:
            body.append(f"Plots saved: {len(plots)} (see paths above)\n", style=f"dim {SUBTEXT}")
        return body

    def eda_review_prompt(self, target, summary, findings, plots) -> str:
        self.stream_end()
        body = self._eda_body(target, summary, findings, plots)
        body.append(
            "\nComment to retarget or adjust analyses, or press Enter to continue.",
            style=f"dim {SUBTEXT}",
        )
        self._console.print(
            Panel(body, title="Review EDA", border_style=S_PANEL_REVIEW, padding=(0, 1))
        )
        return self._pt_input(
            "Comment",
            hint="Enter = continue to features  ·  type to revise EDA  ·  Ctrl-C = abort",
        )

    # ── feature generation ──────────────────────────────────────────────────

    def feat_plan_summary(self, specs: list[dict]) -> None:
        self.stream_end()
        if not specs:
            self._console.print(f"  [{SUBTEXT}]No feature specs proposed.[/{SUBTEXT}]")
            return
        kinds: dict[str, int] = {}
        for s in specs:
            kinds[s.get("kind", "?")] = kinds.get(s.get("kind", "?"), 0) + 1
        breakdown = ", ".join(f"{n} {k}" for k, n in kinds.items())
        t = Text("  ")
        t.append("✓", style=f"bold {GREEN}")
        t.append(f"  Proposed {len(specs)} features:", style=f"dim {GREEN}")
        t.append(f" {breakdown}", style=TEXT)
        self._console.print(t)

    def feat_build_summary(self, meta: dict, build_log: list[dict]) -> None:
        self.stream_end()
        if "error" in meta:
            t = Text("  ")
            t.append("✗", style=f"bold {RED}")
            t.append(f"  Build failed: {meta['error']}", style=TEXT)
            self._console.print(t)
            return
        t = Text("  ")
        t.append("✓", style=f"bold {GREEN}")
        t.append("  Matrix:", style=f"dim {GREEN}")
        t.append(
            f" {meta.get('n_rows', '?'):,} rows × {meta.get('n_features', '?')} features",
            style=TEXT,
        )
        self._console.print(t)
        failed = [e for e in build_log if e.get("applied") is False or e.get("skipped")]
        for e in failed:
            label = e.get("name") or e.get("op") or e.get("method") or "spec"
            why = e.get("error") or e.get("skipped") or ""
            self._console.print(f"    [{PEACH}]⚠ skipped {label}: {why}[/{PEACH}]")

    def feat_scorecard(self, scorecard: dict, metric: dict) -> None:
        self.stream_end()
        if "error" in scorecard and not scorecard.get("features"):
            self._console.print(f"\n  [{RED}]✗[/{RED}] Scoring: {scorecard['error']}")
            return

        mname, mval = metric.get("name"), metric.get("value")
        metric_str = (
            f"{mname}={mval:.4f}"
            if mname and isinstance(mval, (int, float))
            else "(no model metric)"
        )
        leak = scorecard.get("leakage_features", [])
        baseline = Text("\n  ")
        baseline.append("Baseline ", style="bold")
        baseline.append(metric_str, style=f"bold {SAPPHIRE}")
        baseline.append(
            f"  {scorecard.get('n_features')} features, {scorecard.get('n_rows')} rows",
            style=f"dim {SUBTEXT}",
        )
        self._console.print(baseline)
        if leak:
            self._console.print(f"  [{PEACH}]⚠ leakage suspects: {', '.join(leak)}[/{PEACH}]")

        features = scorecard.get("features", [])[:15]
        imp_vals = [
            f.get("importance") for f in features if isinstance(f.get("importance"), (int, float))
        ]
        max_imp = max(imp_vals) if imp_vals else 1.0

        tbl = Table(box=box.SIMPLE_HEAD, padding=(0, 1), show_edge=False, show_header=True)
        tbl.add_column("feature",    style=TEXT,              no_wrap=True, header_style=S_TABLE_HEADER)
        tbl.add_column("importance", justify="right",         no_wrap=True, header_style=S_TABLE_HEADER)
        tbl.add_column("bar",                                 no_wrap=True, header_style=S_TABLE_HEADER)
        tbl.add_column("MI",         justify="right",         no_wrap=True, header_style=S_TABLE_HEADER,
                       style=f"dim {SUBTEXT}")
        tbl.add_column("null %",     justify="right",         no_wrap=True, header_style=S_TABLE_HEADER)
        tbl.add_column("corr",       justify="right",         no_wrap=True, header_style=S_TABLE_HEADER,
                       style=f"dim {SUBTEXT}")

        for f in features:
            imp = f.get("importance")
            imp_str = f"{imp:.3f}" if isinstance(imp, (int, float)) else "—"
            imp_bar = self._importance_bar(imp or 0, max_imp) if isinstance(imp, (int, float)) else ""
            mi = f.get("mutual_info")
            mi_str = f"{mi:.3f}" if isinstance(mi, (int, float)) else "—"
            null_str = f"{f.get('null_rate', 0):.0%}"
            corr = f.get("corr_target")
            corr_str = f"{corr:+.2f}" if isinstance(corr, (int, float)) else "—"
            name = f.get("name", "?")
            if f.get("leakage"):
                name = f"[{PEACH}]{name} ⚠[/{PEACH}]"
            tbl.add_row(name, imp_str, imp_bar, mi_str, null_str, corr_str)

        self._console.print("  ", end="")
        self._console.print(tbl)

        if len(features) >= 3 and imp_vals:
            try:
                import plotext as plt
                plot_w = min(PLOT_WIDTH, self._console.width - 6)
                chart_feats = [f for f in features if isinstance(f.get("importance"), (int, float))]
                names = [f.get("name", "?")[:20] for f in chart_feats]
                values = [f.get("importance") for f in chart_feats]
                if names:
                    plt.clear_figure()
                    plt.bar(names, values, orientation="horizontal", color=PLOT_COLOR_1)
                    plt.title("Feature Importance")
                    plt.plot_size(plot_w, len(names) + 5)
                    plt.theme("dark")
                    self._console.print()
                    plt.show()
            except ImportError:
                pass

    def feat_critique_summary(self, changes: dict, history: list[dict], action: str) -> None:
        self.stream_end()
        changes = changes or {}
        added   = changes.get("added", [])
        dropped = changes.get("dropped", [])
        kept    = changes.get("kept", 0)
        verdict = "refining" if action == "refine" else "converged"
        t = Text("  ")
        t.append("✓", style=f"bold {GREEN}")
        t.append(f"  Critique ({verdict}):", style=f"dim {SUBTEXT}")
        t.append(f" kept {kept}, +{len(added)} added, -{len(dropped)} dropped", style=TEXT)
        self._console.print(t)
        if added:
            self._console.print(f"    [{SUBTEXT}]+ {', '.join(added)}[/{SUBTEXT}]")
        if dropped:
            self._console.print(f"    [{SUBTEXT}]- {', '.join(dropped)}[/{SUBTEXT}]")
        metrics = [h.get("metric") for h in history if h.get("metric") is not None]
        if len(metrics) >= 2:
            delta = metrics[-1] - metrics[-2]
            self._console.print(
                f"    [{SUBTEXT}]metric Δ {delta:+.4f} (now {metrics[-1]})[/{SUBTEXT}]"
            )
            try:
                import plotext as plt
                plot_w = min(max(PLOT_WIDTH // 2, 20), self._console.width - 6)
                plt.clear_figure()
                plt.plot(list(range(1, len(metrics) + 1)), metrics, color=PLOT_COLOR_2)
                plt.title("Metric trajectory")
                plt.xlabel("Iteration")
                plt.plot_size(plot_w, max(PLOT_HEIGHT // 2, 5))
                plt.theme("dark")
                self._console.print()
                plt.show()
            except ImportError:
                pass
        elif metrics:
            self._console.print(f"    [{SUBTEXT}]metric: {metrics[-1]}[/{SUBTEXT}]")

    def feat_review_prompt(self, meta, scorecard, metric, summary, history) -> str:
        self.stream_end()
        body = Text()
        body.append(f"{summary}\n", style=TEXT)
        cols = (meta or {}).get("columns", [])
        if cols:
            shown = ", ".join(cols[:20]) + (" …" if len(cols) > 20 else "")
            body.append(f"\nFeatures ({len(cols)}): ", style=f"bold {TEXT}")
            body.append(f"{shown}\n", style=TEXT)
        leak = (scorecard or {}).get("leakage_features", [])
        if leak:
            body.append(f"\n⚠ leakage suspects: {', '.join(leak)}\n", style=PEACH)
        if history:
            traj = " → ".join(str(h.get("metric")) for h in history)
            body.append(f"\nMetric trajectory: {traj}\n", style=f"dim {SUBTEXT}")
        body.append(
            "\nComment to adjust features (add/drop/iterate), or press Enter to continue.",
            style=f"dim {SUBTEXT}",
        )
        self._console.print(
            Panel(body, title="Review features", border_style=S_PANEL_REVIEW, padding=(0, 1))
        )
        return self._pt_input(
            "Comment",
            hint="Enter = finalize features  ·  type to iterate  ·  Ctrl-C = abort",
        )

    # ── spinner ───────────────────────────────────────────────────────────

    @contextmanager
    def spinner(self, label: str) -> Generator[None, None, None]:
        with self._console.status(
            f"[{MAUVE}]{label}[/{MAUVE}]",
            spinner="dots",
            spinner_style=f"bold {MAUVE}",
        ):
            yield

    # ── error ─────────────────────────────────────────────────────────────

    def error(self, msg: str) -> None:
        self.stream_end()
        t = Text("\n  ")
        t.append("✗ Error:", style=f"bold {RED}")
        t.append(f"  {msg}", style=TEXT)
        self._console.print(t)
