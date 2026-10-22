"""Runtime schema: optional YAML, env/.env for secrets and overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    anthropic_model: str = "claude-opus-4-7"
    openai_model: str = "gpt-5.1"
    llm_temperature: float = 0.0
    # Provider-neutral switch: surface model reasoning in traces when supported.
    llm_reasoning: bool = True

    # --- Data source ---
    # Postgres now; same URL slot reused per-source later (e.g. snowflake://...).
    database_url: str | None = Field(
        default=None,
        description="SQLAlchemy URL, e.g. postgresql+psycopg://user:pass@host:5432/db",
    )
    sample_size: int = 10_000
    # Hard cap on rows any single read may return into memory.
    max_result_rows: int = 10_000_000

    # --- EDA / artifacts ---
    # Base directory for run artifacts (plots, exports). Per-run subdir uses run_id.
    artifacts_dir: str = "artifacts"
    # Cap on categories/columns shown in categorical plots and frequency tables.
    eda_top_k: int = 10
    # |correlation| at or above this is flagged as a notable pair.
    eda_corr_threshold: float = 0.7
    # Cap on PNGs rendered per EDA pass (only finding-cited plots are saved).
    eda_max_plots: int = 12

    # --- Feature generation ---
    # Max refinement passes the agentic feature loop may run.
    feat_max_iterations: int = 4
    # Min baseline-metric gain to justify another refinement pass (else converge).
    feat_metric_epsilon: float = 0.005

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSource(
                settings_cls=settings_cls,
                yaml_file=Path(os.environ.get("CONFIG_FILE") or "datasynth.yaml"),
            ),
        )
