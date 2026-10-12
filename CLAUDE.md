# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

SMU final project: a demo stock screener that turns social-media and news chatter into a weekly Top 5 of US stocks. It is a demo, so keep costs at zero apart from the LLM: FinBERT (ProsusAI/finbert) for sentiment, yfinance for market data, free-tier Finnhub. Do not introduce other paid APIs or services. The LLM is `gpt-4o-mini` via the apiyi.com OpenAI-compatible relay (switched from Groq on 2026-10-08) on a key with a **$1 hard limit**; `run_pipeline.py` prints spend vs limit after each run (`llm_budget()`), warns at 80%, and the app shows a banner. Tell the user when spend is getting close to the limit. Outputs carry a "not investment advice" disclaimer; keep it.

## Environment and commands

Always run commands from `Final_Project_CJB/` using the `CJB` conda env (`conda activate CJB`, or `/opt/anaconda3/envs/CJB/bin/python`). The system `python3` lacks the dependencies. Modules use package imports (`from filtering_of_data import ...`) that resolve from this folder.

```bash
python run_pipeline.py --daily          # collect data + Phase 1 only: builds z-score history (phase1_history)
python run_pipeline.py                  # weekly: collect + Phases 1-5, publishes the Top 5
python run_pipeline.py --skip-collect   # weekly run on data already in the DB (no collection API calls)

python -m streamlit run frontend/app.py # frontend (reads latest.json directly; API not required)
uvicorn api:app --reload --port 8000    # optional read-only HTTP API (/docs for endpoints)

python data_base_creation/trigger.py [--rebuild|--skip-collect|--refresh-universe]  # data stage only
python filtering_of_data/phaseN_filter.py   # run one phase standalone (each runs earlier phases first)
python filtering_of_data/phase5_filter.py   # uses saved Phase 4 output; --fresh reruns 1-4
```

There is no test suite or linter config. Phase files can be run individually to debug one stage at a time.

## Architecture

Three layers, communicating through SQLite and a JSON file:

1. **Data collection** (`data_base_creation/`): `trigger.py` runs the StockTwits, Finnhub and RSS collectors, then `categorizer.py` turns raw rows into `ProcessedNews` (one row per post×ticker). `us_universe.py` defines which tickers count (US common stocks/REITs/ADRs on NYSE/Nasdaq/NYSE American, cached 7 days from one Finnhub call); ETFs and OTC names are excluded everywhere. Models and in-place schema upgrades live in `database.py`; the DB is `data_base_creation/market_data.db` (override with `MARKET_DB_PATH`).

2. **Screening funnel** (`filtering_of_data/phase1..5_filter.py`), each phase consuming the previous one's DataFrames:
   - P1: bot elimination + ticker extraction; ranks by `chatter_score` and appends a snapshot to `phase1_history` every run.
   - P2: hard filters (market cap/penny stocks, chatter z-score vs real volume, distinct authors, fundamentals). The z-score uses per-ticker history once enough daily snapshots exist, otherwise falls back to cross-sectional mode, which is why `--daily` runs matter.
   - P3: FinBERT + slang dictionary (seed + learned, `learned_slang.csv`) sentiment with per-ticker clause isolation.
   - P4: LLM catalyst validation and sarcasm review → shortlist. LLM errors are never treated as approval; answers are cached in `llm_cache.json`. Runs on rules alone if no LLM is configured.
   - P5: risk checks (overextension, volatility, earnings, short interest), sector/correlation diversification, and pick tracking vs SPY (`PickHistory`).
   Each phase file has a fallback `from phaseN_filter import ...` so it also runs from inside `filtering_of_data/`.

3. **Publishing and presentation**: `run_pipeline.py` orchestrates everything, records each run in the `pipeline_runs` table, and writes `filtering_of_data/output/latest.json` (plus `run_<id>.json`) containing picks, sources, funnel counts, methodology, reliability and limitations. `frontend/app.py` (Streamlit) and `api.py` (FastAPI) only **read** that result. Change the result schema in `build_summary`/`pick_record` and update both readers together. The result's `dropped` list (built by `build_dropped` from `p2.LAST_REJECTED`, Phase 4 eligibility and `p5.LAST_RESULT`) records why every mentioned stock that isn't a pick was left out; the chat looks up stocks the user names (`find_dropped`) so it can explain drops, which is a core goal of the app.

## Configuration

Secrets live in `data_base_creation/.env` (loaded explicitly by path in most modules): `FINNHUB_API_KEY`, `LLM_PROVIDER` (`openai`), `LLM_API_KEY`, `LLM_MODEL` (`gpt-4o-mini`), `LLM_BASE_URL` (`https://api.apiyi.com/v1`), optional `ALPHAVANTAGE_API_KEY` (Phase 2 skips it if unset). The LLM client is OpenAI-compatible (`openai` package with a provider base URL). The deployed Streamlit app can use `st.secrets` instead. Tunable thresholds are module-level constants at the top of each phase file.
