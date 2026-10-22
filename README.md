# datasynth

An agentic AI CLI that connects to an enterprise database, understands your data, and builds a trained machine learning model — all from a single natural-language ask. No data science expertise required.

---

## The Problem

Enterprise teams sit on structured data that could answer real business questions — churn, fraud risk, demand forecasting, anomaly detection — but extracting that value today requires a data scientist to spend days profiling data, cleaning it, engineering features, running experiments, and writing up results. Most orgs don't have enough of those people.

## The Proposal

datasynth turns that work into a conversation. A business analyst types a question — *"Which customers are most likely to churn next month?"* — and the agent handles everything: connecting to the database, understanding the schema, profiling and cleaning the data, generating features, training a model, and delivering results with explanations a non-technical person can act on.

The key design bet: **the agent thinks out loud**. Every decision — which tables to join, why it dropped a column, what features it invented — is visible to the user in real time. The user can approve, redirect, or override at any step.

---

## How It Works

```
$ uv run main.py

  datasynth  ·  brave-salmon  ·  anthropic

  "Which customers are most likely to churn in the next 30 days?"

→ [connect]   Selecting relevant tables…
→ [profile]   Profiling selected tables…
→ [clean]     Proposing data cleaning…       ← pauses for approval
→ [target]    Defining the prediction target… ← pauses for approval
→ [eda]       Running analyses and plots…    ← pauses for approval
→ [features]  Building feature matrix…       ← pauses for approval

  Artifacts saved to artifacts/brave-salmon/
```

Every reasoning step streams to the terminal before the action runs. The user sees why each decision was made. At each approval gate, press Enter to proceed or type a comment to redirect.

---

## Run Management

Each run gets a human-readable name (e.g. `brave-salmon`, `swift-cedar`) and is persisted to disk. If you interrupt a run partway through, you can pick it up later exactly where you left off — no re-running connect, profile, or cleaning.

```bash
uv run main.py                        # new run, auto-generated name
uv run main.py --name my-experiment   # new run, custom name
uv run main.py --list                 # show all runs with status
uv run main.py --resume brave-salmon  # continue an interrupted run
```

`--list` output:

```
  NAME             STATUS       CREATED              ASK
  brave-salmon     interrupted  2026-06-03 14:23:01  Which customers are most likely to churn…
  swift-cedar      completed    2026-06-02 09:15:43  What drives revenue?
```

Run state is persisted via:
- `artifacts/checkpoints.db` — SQLite graph checkpoint (via LangGraph `AsyncSqliteSaver`)
- `artifacts/<run-name>/frames/` — Parquet snapshots of sampled/cleaned tables
- `artifacts/runs.json` — run registry (name, ask, created_at, status)

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  CLI  (streaming traces · approval gates · --resume) │
├──────────────────────────────────────────────────┤
│  LangGraph orchestrator (AsyncSqliteSaver)        │
├────────────┬─────────────────────────────────────┤
│  LLM layer │  Phase harnesses (scoped tool sets)  │
│  Anthropic │   connect → profile → clean         │
│  OpenAI    │   → target → EDA → feature-gen      │
│  (config)  │   → model → explain → report        │
├────────────┴─────────────────────────────────────┤
│  Tool layer  (deterministic · logged · read-only) │
│  SQLAlchemy · polars · sklearn (baseline today)   │
│  AutoGluon · SHAP (planned)                       │
├──────────────────────────────────────────────────┤
│  Artifact store  (checkpoints · frames · plots)   │
└──────────────────────────────────────────────────┘
```

### Pipeline phases

| Phase | What the agent decides | What runs |
|---|---|---|
| **Connect** | Which tables are relevant to the ask | Schema introspection, FK graph |
| **Profile** | What to measure | Row counts, null rates, dtypes, cardinality |
| **Clean** | Impute / drop / cast strategy | Proposes ops → user approves → applies to `FrameStore` |
| **Target** | Which column/derivation defines the label | SQL derivation or existing column selection |
| **EDA** | Hypotheses worth testing | Correlations, group-bys, outlier scans, matplotlib/seaborn plots |
| **Feature generation** | What signals to create | FK-guided SQL aggregates → RandomForest baseline scorecard → critique loop |
| **Model training** | Task type, target column, eval metric | AutoGluon Tabular *(planned)* |
| **Explain / Report** | What drives predictions | SHAP values + narrative *(planned)* |

### Transparency layer

Every phase streams the agent's reasoning before acting:

- *"Dropping `legacy_id`: 94% null, no FK references, no signal."*
- *"Joining `customers` via `orders.customer_id → customers.id` — adds tenure and region features."*

Approval gates surface the proposed operation before execution. The graph checkpoint survives process exit — resuming skips already-completed phases.

---

## Implementation status

What runs today when you execute `uv run main.py`:

| Phase | Status | Notes |
|---|---|---|
| **Connect** | ✅ | LLM table selection from schema + FKs; hallucinated names dropped |
| **Profile** | ✅ | Per-table samples, null rates, cardinality, quality flags; Object dtypes auto-coerced to String |
| **Clean** | ✅ | Agent proposes ops from a fixed vocabulary; user approves via interrupt; applies to `FrameStore` |
| **Target definition** | ✅ | LLM proposes label column or SQL derivation; user approves |
| **EDA** | ✅ | Agent plans analyses; deterministic runner + matplotlib/seaborn plots saved under `artifacts/<run-name>/eda/` |
| **Feature generation** | ✅ | Agentic plan → FK-guided SQL build → sklearn RF baseline scorecard → critique loop |
| **Run persistence / resume** | ✅ | `AsyncSqliteSaver` checkpoints + Parquet frame snapshots; `--resume`, `--list`, `--name` flags |
| **Knowledge graph** | ⏳ | Planned (networkx); FK paths used today inside feature SQL only |
| **Model training** | ⏳ | Planned (AutoGluon Tabular) |
| **Explain / Report** | ⏳ | Planned (SHAP + narrative + pipeline export) |

---

## Repository layout

```
datasynth/
  config.py              # Pydantic-settings: env / .env / datasynth.yaml
  runs.py                # Run registry + human-readable name generation
  _warnings.py           # Suppress noisy third-party warnings
  datasource/
    base.py              # DataSource ABC + SchemaInfo / TableInfo / ForeignKey
    readonly.py          # assert_readonly() SQL guard (defense-in-depth)
    postgres.py          # Read-only Postgres implementation
  llm/
    provider.py          # get_chat_model() → LangChain BaseChatModel (Anthropic | OpenAI)
  graph/
    graph.py             # LangGraph compile: connect → … → feature review
    state.py             # AgentState (serializable snapshots; DataSource/FrameStore via config)
    review.py            # Reusable interrupt() review / revise gate
    nodes/               # connect, profile, clean, eda, features, target, select_review
  clean/
    ops.py               # Deterministic cleaning vocabulary + apply_ops
    store.py             # FrameStore: in-memory working copies + Parquet persistence
  eda/
    analyses.py          # Deterministic EDA vocabulary + run_analyses
    plots.py             # Plot helpers (saved via ArtifactStore)
    render.py            # EDA finding renderers
  features/
    sql_build.py         # Whitelist-validated SQL for entity-grain feature matrix
    evaluate.py          # Baseline RandomForest scorecard for the refinement loop
    transforms.py        # Feature transform helpers
    vocabulary.py        # Feature op vocabulary
  artifacts/
    store.py             # Per-run directory layout under artifacts/<run-name>/
  cli/
    runner.py            # astream_events driver + interrupt resume loop
    renderer.py          # Rich terminal UI (phases, thinking stream, review prompts)
main.py                  # Entrypoint: one ask → full graph + --resume / --list / --name
pyproject.toml
datasynth.yaml           # Optional non-secret config (see datasynth.example.yaml)
.env                     # Secrets: API keys, database URL (gitignored)
dev-db/                  # Docker Postgres + synthetic B2B SaaS churn dataset
artifacts/
  checkpoints.db         # SQLite graph checkpoints (all runs)
  runs.json              # Run registry (name, ask, status, timestamps)
  <run-name>/
    frames/              # Parquet snapshots of sampled/cleaned tables
    eda/                 # EDA plots (PNG)
```

---

## Configuration

Config is layered (highest priority first): constructor args → environment variables → `.env` file → `datasynth.yaml`.

**`.env`** — secrets, never commit:
```env
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/mydb
```

**`datasynth.yaml`** — non-secret runtime settings (see `datasynth.example.yaml` for the full list):
```yaml
llm_provider: anthropic          # anthropic | openai
anthropic_model: claude-opus-4-7
llm_reasoning: true              # surface model reasoning in traces
sample_size: 10000
max_result_rows: 10000000
artifacts_dir: artifacts         # run subdirs: artifacts/<run-name>/eda/ etc.
eda_top_k: 10
eda_corr_threshold: 0.7
feat_max_iterations: 4
feat_metric_epsilon: 0.005
```

### Supported LLM providers

| Provider | Env var | Default model | Reasoning |
|---|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-4-7` | Extended thinking (`budget_tokens=4096`) |
| OpenAI | `OPENAI_API_KEY` | `gpt-5.1` | `reasoning_effort: medium` |

---

## Data safety

datasynth is strictly read-only against the source database. Three layers enforce this:

1. **Database role** *(primary control)* — connect with a role that has only `SELECT` / `USAGE` privileges. Do not give datasynth a write-capable credential.
2. **Statement guard** — `assert_readonly()` rejects any query that does not start with `SELECT` or `WITH`, or contains write keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.).
3. **Transaction-level read-only** — every query executes inside `SET TRANSACTION READ ONLY`, so the server refuses writes even if the guard is bypassed.

The agent only ever sees schema metadata and sampled rows — it never loads an entire table into LLM context. Cleaning ops mutate only the local `FrameStore` (in-memory + Parquet snapshots); the source database is never touched.

---

## Getting started

**Prerequisites:** Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd datasynth
uv sync
cp .env.example .env        # fill in API key + database URL
uv run main.py              # start a new run
```

### Local dev database

For development without a production database, use the synthetic B2B SaaS churn dataset in [`dev-db/`](dev-db/):

```bash
cd dev-db && ./up.sh      # start Postgres + reseed (destructive wipe)
cd dev-db && ./down.sh    # stop containers (keeps data volume)
cd dev-db && ./down.sh -v # stop and delete volume (full reset)
```

Starts Postgres on **localhost:5433**. Copy `DATABASE_URL` into the project root `.env`:

```env
DATABASE_URL=postgresql+psycopg://datasynth_ro:datasynth_ro_dev@localhost:5433/churn_dev
```

Then run `uv run main.py` from the repo root. The agent runs through table selection → profiling → optional cleaning → target definition → EDA (plots under `artifacts/<run-name>/eda/`) → feature generation, with approval gates at each step.

Interrupt at any point with `Ctrl+C`. Resume later with `uv run main.py --resume <name>` — the graph picks up exactly where it left off, skipping all completed phases.

---

## Roadmap

### Phase 0 — Foundation ✅
- [x] `DataSource` ABC + Postgres read-only implementation (FK introspection, sample, cost estimate)
- [x] Provider-neutral LLM factory (Anthropic / OpenAI, unified `BaseChatModel`)
- [x] Layered config (env / `.env` / YAML)
- [x] Read-only SQL guard (3-layer enforcement)

### Phase 1 — Agent skeleton ✅
- [x] LangGraph state model + `AsyncSqliteSaver` checkpointer (durable, per-run `thread_id`)
- [x] `interrupt()` approval gates (table selection, clean, target, EDA, features)
- [x] Streaming trace renderer (`cli/renderer.py` + `astream_events`)
- [x] Connect + Profile phase nodes
- [x] Human-readable run names + registry (`--list`, `--resume`, `--name`)
- [x] `FrameStore` Parquet persistence for cross-session resume

### Phase 2 — EDA + Clean ✅
- [x] EDA: vocabulary-driven analyses, plots, synthesis, review gate
- [x] Clean: agent-proposed ops, review gate, `FrameStore` apply + re-profile
- [x] Target definition: LLM proposes label column or SQL derivation
- [x] Artifact store for EDA plots (`artifacts/<run-name>/eda/`)

### Phase 3 — Feature generation ✅
- [x] FK-guided feature SQL (`features/sql_build.py`) at entity grain
- [x] Feature-gen agentic loop (plan → build → evaluate → critique → review)
- [ ] Knowledge graph module (networkx) + path explanations for business users

### Phase 4 — Model training + Explain
- [ ] AutoGluon Tabular integration (task-type detection, metric selection, leaderboard)
- [ ] SHAP explanations + narrative summary
- [ ] Reproducible pipeline script export

### Phase 5 — Polish + Enterprise
- [ ] Snowflake connector (pushdown cost estimates)
- [ ] Data-mined relationships (correlation, mutual information, entity resolution)
- [ ] Full REPL with history and session management
- [ ] Auth / credential management for multi-user deployments

---

## Design principles

**Agent decides, tools execute.** The LLM reasons about *what* to do. Deterministic, logged Python functions do *how*. This keeps the pipeline reproducible and auditable even though the planner is non-deterministic.

**Transparent by default.** Every agent decision streams to the terminal before it acts. Users should never feel the system is a black box, even if they don't understand the ML.

**Approval gates on anything expensive or irreversible.** The agent proposes; the user confirms. This applies to any operation that takes significant time, scans large data, or changes the working dataset.

**Read-only at the source.** datasynth never writes to the database it connects to. All mutations happen in the local artifact store.

**Resumable by default.** Every run is checkpointed after each graph step. Interrupting and resuming costs nothing — no re-running phases the agent already completed.

**Backend-neutral from day one.** The `DataSource` ABC means swapping Postgres for Snowflake (or adding both) is one new class, not a refactor.
