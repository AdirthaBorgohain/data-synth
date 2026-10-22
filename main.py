"""CLI entrypoint — thin async shell.

Prompts the user for a question, wires up the DataSource and Renderer,
then hands off to cli.runner which drives the LangGraph pipeline.
Replaced by datasynth/cli/ REPL once the full agent layers land.

Usage:
  uv run main.py                        # new run, auto-generated name
  uv run main.py --name my-run          # new run, custom name
  uv run main.py --resume brave-salmon  # continue an interrupted run
  uv run main.py --list                 # list all runs
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
from pathlib import Path

from datasynth._warnings import configure_warnings

configure_warnings()

from rich.console import Console
from rich.table import Table

from datasynth.cli.renderer import Renderer
from datasynth.cli import runner
from datasynth.clean.store import FrameStore
from datasynth.config import Settings
from datasynth.datasource.postgres import PostgresDataSource
from datasynth.runs import (
    append_run,
    find_run,
    load_registry,
    new_name,
    update_status,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="datasynth", add_help=True)
    parser.add_argument("--resume", metavar="NAME", help="resume an interrupted run by name")
    parser.add_argument("--name", metavar="NAME", help="custom name for a new run (auto-generated if omitted)")
    parser.add_argument("--list", action="store_true", help="list all runs and exit")
    return parser.parse_args()


def _print_runs(records: list[dict]) -> None:
    console = Console()
    if not records:
        console.print("  No runs found.", style="dim")
        return
    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("NAME", style="cyan")
    t.add_column("STATUS")
    t.add_column("CREATED")
    t.add_column("ASK")
    status_styles = {"completed": "green", "interrupted": "yellow", "running": "blue"}
    for r in reversed(records):
        status = r.get("status", "?")
        ask = r.get("ask", "")
        if len(ask) > 60:
            ask = ask[:57] + "..."
        created = r.get("created_at", "")[:19].replace("T", " ")
        t.add_row(r["name"], f"[{status_styles.get(status, 'white')}]{status}[/]", created, ask)
    console.print()
    console.print(t)
    console.print()


async def _prompt_ask() -> str:
    try:
        import concurrent.futures

        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.styles import Style

        from datasynth.cli.theme import PROMPT_STYLE_PAIRS

        _pt_style = Style.from_dict(dict(PROMPT_STYLE_PAIRS))
        _pt_toolbar = HTML('<bottom-toolbar>  datasynth · agentic ML pipeline  </bottom-toolbar>')
        _pt_prompt = HTML(
            '<prompt>  What would you like to predict or understand?'
            '\n  </prompt><hint>&#8250;</hint> '
        )

        def _ask() -> str:
            try:
                return pt_prompt(
                    _pt_prompt,
                    style=_pt_style,
                    bottom_toolbar=_pt_toolbar,
                ).strip()
            except (EOFError, KeyboardInterrupt):
                return ""

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(_ask).result()
        except RuntimeError:
            return _ask()

    except ImportError:
        try:
            return input("  What would you like to predict or understand?\n  › ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""


async def _main() -> None:
    args = _parse_args()
    settings = Settings()
    renderer = Renderer()

    artifacts_base = Path(settings.artifacts_dir)
    artifacts_base.mkdir(parents=True, exist_ok=True)
    registry_path = artifacts_base / "runs.json"
    checkpoint_db = artifacts_base / "checkpoints.db"

    if args.list:
        _print_runs(load_registry(registry_path))
        return

    if not settings.database_url:
        renderer.error("DATABASE_URL not set. Add it to .env and rerun.")
        sys.exit(1)

    ds = PostgresDataSource(
        settings.database_url,
        default_sample_size=settings.sample_size,
        max_result_rows=settings.max_result_rows,
    )
    connected, db_name = ds.probe_connection()
    renderer.welcome(
        data_source_connected=connected,
        data_source_db=db_name,
    )

    try:
        if args.resume:
            record = find_run(registry_path, args.resume)
            if record is None:
                renderer.error(
                    f"Run '{args.resume}' not found. "
                    "Use --list to see available runs."
                )
                return
            run_id = record["name"]
            ask = record["ask"]
            frames_dir = artifacts_base / run_id / "frames"
            frame_store = FrameStore.load(frames_dir)
            is_resume = True
        else:
            ask = await _prompt_ask()
            if not ask:
                return

            if args.name:
                run_name = args.name.strip()
                if not run_name or " " in run_name:
                    renderer.error("Run name must be non-empty and contain no spaces.")
                    return
                if find_run(registry_path, run_name):
                    renderer.error(
                        f"Run '{run_name}' already exists. "
                        "Choose a different name or use --resume to continue it."
                    )
                    return
            else:
                run_name = new_name(registry_path)

            run_id = run_name
            frame_store = None
            is_resume = False

        renderer.run_header(run_id, ask, settings.llm_provider)

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db)) as checkpointer:
            await checkpointer.setup()

            if not is_resume:
                append_run(registry_path, {
                    "name": run_id,
                    "ask": ask,
                    "created_at": datetime.datetime.now().isoformat(),
                    "status": "running",
                })
            else:
                update_status(registry_path, run_id, "running")

            try:
                await runner.run(
                    ask, run_id, settings, ds, renderer,
                    checkpointer=checkpointer,
                    frame_store=frame_store,
                    is_resume=is_resume,
                )
                update_status(registry_path, run_id, "completed")
            except KeyboardInterrupt:
                update_status(registry_path, run_id, "interrupted")
                renderer.error("Interrupted.")
            except Exception as exc:
                update_status(registry_path, run_id, "interrupted")
                renderer.error(str(exc))
                raise
    finally:
        ds.close()


if __name__ == "__main__":
    asyncio.run(_main())
