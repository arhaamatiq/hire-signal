# 🧠 HireSignal

> Decodes AI lab hiring patterns into actionable intelligence — who's scaling, what domains they're investing in, and what's coming next.

## What it does

HireSignal is an agentic LangGraph pipeline that:

1. **Searches** the web for current AI/ML job postings across any set of companies
2. **Extracts** skills, domains, and role patterns from raw posting data
3. **Computes** week-over-week hiring velocity deltas against historical snapshots
4. **Scores** each company 0–100 on AI intent (hiring surge + domain signal + funding news)
5. **Synthesizes** a ranked intelligence brief via LLM — who's scaling, what they're building, what to watch

## Stack

- **LangGraph** — agentic stateful pipeline with parallel company research via `Send()`
- **LangChain + Tavily** — real-time web search for job postings and company news
- **Groq / OpenAI** — LLM synthesis (llama-3.3-70b-versatile or gpt-4o-mini)
- **FastAPI** — REST API with `POST /analyze`
- **Python 3.11+ / uv** — dependency management

## Project structure

```
hiresignal/
├── frontend/
│   └── app.py                # Streamlit SaaS UI — run analysis, view intent scores & report
├── agent/
│   └── agentic_workflow.py   # LangGraph pipeline (GraphBuilder, 7 nodes, state)
├── tools/
│   ├── job_search_tool.py    # @tool wrappers for JobFetcher
│   ├── company_news_tool.py  # @tool wrappers for NewsFetcher
│   ├── skill_extractor_tool.py
│   └── trend_delta_tool.py
├── utils/
│   ├── config_loader.py
│   ├── model_loader.py
│   ├── skill_taxonomy.py     # SKILL_DOMAIN_MAP, compute_intent_score()
│   ├── trend_calculator.py   # TrendCalculator.compute_delta()
│   ├── data_store.py         # JSON snapshot persistence
│   ├── job_fetcher.py        # Tavily job search
│   ├── news_fetcher.py       # Tavily news + funding search
│   └── report_exporter.py   # Markdown report export
├── prompt_library/
│   └── prompts.py            # NARRATIVE_PROMPT, ORCHESTRATOR_PROMPT
├── config/
│   └── config.yaml
├── main.py                   # FastAPI app
└── pyproject.toml
```

## Quickstart

```bash
# 1. Clone and install (Python 3.11+ required)
git clone https://github.com/yourname/hiresignal
cd hiresignal
uv sync

# 2. Set API keys
cp .env.example .env
# Edit .env — add GROQ_API_KEY, TAVILY_API_KEY (and OPENAI_API_KEY if using model_provider=openai)

# 3. Run the API (terminal 1)
uv run uvicorn main:app --reload

# 4. Run the Streamlit frontend (terminal 2)
uv run streamlit run frontend/app.py

# Open http://localhost:8501 — select companies, set lookback days, and click "Run analysis".
```

**API-only (no UI):**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"companies": ["Anthropic", "OpenAI", "Mistral"], "timeframe_days": 30}'
```

**Alternative (pip):** if you don't use `uv`, create a venv and install in editable mode:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt -e .
.venv/bin/uvicorn main:app --reload
```

## API

### `POST /analyze`

```json
{
  "companies": ["Anthropic", "OpenAI", "Google DeepMind"],
  "timeframe_days": 30,
  "model_provider": "groq"
}
```

Response:
```json
{
  "final_report": "# HireSignal Intelligence Brief...",
  "companies_analyzed": ["Anthropic", "OpenAI", "Google DeepMind"],
  "intent_scores": {"Anthropic": 78, "OpenAI": 65, "Google DeepMind": 71},
  "report_filepath": "./output/hiresignal_2025-03-10_14-22-00.md"
}
```

### `GET /health`

```json
{"status": "ok", "pipeline": "compiled"}
```

## Streamlit frontend

The `frontend/` app provides a modern SaaS-style UI:

- **Company multiselect** — pick from preset AI/ML companies (defaults from `config.yaml`)
- **Lookback slider** — 7–90 days for posting velocity
- **Model provider** — Groq or OpenAI (sidebar)
- **API URL** — configurable backend base URL (default `http://localhost:8000`)
- **Intent score cards** — ranked leaderboard with High / Medium / Low–medium / Low labels
- **Intelligence brief** — full markdown report rendered in-app; reports are also saved to `output/`

Run with: `uv run streamlit run frontend/app.py` (ensure the API is running first).

## How the pipeline works

```
validate_input → coordinator → company_researcher ×N (parallel) → extract_skills → analyze_trends → generate_narratives → synthesize_report
```

`company_researcher` runs in parallel for each company via LangGraph's `Send()` API. All parallel results are merged back into shared state via the `merge_dicts` reducer before downstream nodes execute.

## Intent score formula

| Component | Weight |
|-----------|--------|
| Base | 40 |
| Posting velocity delta | ±30 |
| High-signal domain coverage | +20 |
| Funding / partnership news | +10 |

Score interpretation: 75+ = HIGH, 55–74 = MEDIUM, 35–54 = LOW-MEDIUM, <35 = LOW
