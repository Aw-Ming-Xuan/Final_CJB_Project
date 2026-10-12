"""One-command data stage: collect from all sources, then categorize.

    python data_base_creation/trigger.py              # normal run
    python data_base_creation/trigger.py --rebuild    # wipe processed_news and redo it
    python data_base_creation/trigger.py --skip-collect   # categorize only, no API calls
    python data_base_creation/trigger.py --refresh-universe   # re-download the US stock list

Changes from the first version:
- Project root is found from this file's location (no hardcoded path).
- A missing FINNHUB_API_KEY no longer kills the whole run (the collector raises
  SystemExit, which `except Exception` does not catch).
- The database session is opened AFTER collection, not before.
- --rebuild / --skip-collect flags.
- The summary now shows row counts, the most-discussed tickers and a data
  health check, instead of just the last 10 rows.
"""

import argparse
import os
import sys

# This file lives in <root>/data_base_creation/, so the project root is one level up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from data_base_creation.database import SessionLocal, engine, init_db
from data_base_creation.news_collector import fetch_news
from data_base_creation.finnhub_news_collector import fetch_news_api
from data_base_creation.stocktwits_collector import fetch_stocktwits
from data_base_creation.categorizer import process_unassigned_data
from data_base_creation.us_universe import refresh_us_universe, load_us_symbols


def run_step(label, func):
    """Run one collector; a failure in one source never stops the others."""
    print(f"\n--- {label} ---")
    try:
        func()
    except (Exception, SystemExit) as e:   # SystemExit: missing API key
        print(f"  {label} failed: {e}")


def print_summary():
    print("\n=== Summary of Stored & Categorized Entries ===")

    counts = pd.read_sql(
        """
        SELECT
          (SELECT COUNT(*) FROM raw_news_posts)  AS news_rows,
          (SELECT COUNT(*) FROM stocktwits_posts) AS stocktwits_rows,
          (SELECT COUNT(*) FROM processed_news)   AS processed_rows
        """,
        con=engine,
    )
    print(counts.to_string(index=False))

    top = pd.read_sql(
        """
        SELECT ticker,
               COUNT(DISTINCT raw_post_id)                    AS posts,
               SUM(CASE WHEN source_type = 'SOCIAL' THEN 1 ELSE 0 END) AS social,
               SUM(CASE WHEN source_type = 'NEWS'   THEN 1 ELSE 0 END) AS news,
               ROUND(AVG(sentiment_score), 2)                 AS avg_sentiment
        FROM processed_news
        WHERE ticker != 'MARKET'
        GROUP BY ticker
        ORDER BY posts DESC
        LIMIT 15
        """,
        con=engine,
    )
    print("\nMost-discussed tickers:")
    print(top.to_string(index=False) if not top.empty else "  (none yet)")

    # Data health checks
    health = pd.read_sql(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN author_join_date IS NULL THEN 1 ELSE 0 END) AS missing_join_date
        FROM stocktwits_posts
        """,
        con=engine,
    )
    total = int(health["total"][0] or 0)
    missing = int(health["missing_join_date"][0] or 0)
    print("\nData health:")
    if total == 0:
        print("  WARNING: no StockTwits posts stored. Phase 2 will have no social data.")
    elif missing / total > 0.5:
        print(f"  WARNING: {missing}/{total} StockTwits posts have no author join date "
              "(the bot filter can't work; check parse_join_date).")
    else:
        print(f"  StockTwits author data OK ({total - missing}/{total} have a join date).")


def refresh_universe(force=False):
    """Download the US stock list (one Finnhub call, cached for 7 days)."""
    print("\n--- US Market Universe ---")
    session = SessionLocal()
    try:
        refresh_us_universe(session, force=force)
        print(f"  US common stocks available: {len(load_us_symbols(session))}")
    finally:
        session.close()


def run_pipeline(rebuild=False, skip_collect=False, refresh_universe_now=False):
    print("=== Step 1: Initializing / Upgrading Database ===")
    init_db()
    try:
        refresh_universe(force=refresh_universe_now)
    except (Exception, SystemExit) as e:
        print(f"  US universe step failed: {e}")

    if not skip_collect:
        print("\n=== Step 2: Collecting Data from Sources ===")
        # StockTwits first: the RSS (Nasdaq per-ticker) and Finnhub collectors use its trending tickers
        run_step("StockTwits Collector", fetch_stocktwits)
        run_step("RSS News Collector", fetch_news)
        run_step("Finnhub API Collector", fetch_news_api)
    else:
        print("\n=== Step 2: Skipped (--skip-collect) ===")

    print("\n=== Step 3: Categorizing & Analyzing Raw Data ===")
    session = SessionLocal()
    try:
        count = process_unassigned_data(session, rebuild=rebuild)
        print(f"Successfully processed and categorized {count} new records!")
    finally:
        session.close()

    print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and categorize market chatter.")
    parser.add_argument("--rebuild", action="store_true",
                        help="wipe processed_news and re-categorize everything")
    parser.add_argument("--skip-collect", action="store_true",
                        help="skip the collectors, only categorize what is already stored")
    parser.add_argument("--refresh-universe", action="store_true",
                        help="force a fresh download of the US stock list")
    args = parser.parse_args()
    run_pipeline(rebuild=args.rebuild, skip_collect=args.skip_collect,
                 refresh_universe_now=args.refresh_universe)
